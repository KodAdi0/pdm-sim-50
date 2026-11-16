"""
Filo ölçeğinde koltuk aktüatörü kestirimci bakım — sentetik veri simülatörü.

Genel (ürüne özgü olmayan) bir business class koltuk hareket sistemini modeller:
  koltuk başına 1 kontrol ünitesi (ECU) + 2 aktüatör (TRK: kaydırma, PRV: gizlilik paneli).

Katmanlar
  1. Filo ve uçuş planı (uçak profili, günlük uçuş, doluluk, bakım günleri)
  2. Kullanım (koltuk başına günlük hareket sayısı)
  3. Gizli fiziksel bozulma (Weibull ömür, arıza moduna göre bozulma eğrisi)
  4. Kontrol ünitesi davranışı (eşik tabanlı olay logu + bağlam, sayaçlar, kritik arıza kodları, IBITE)
  5. Bakım (arıza sonrası değişim, şikâyete bağlı erken söküm, atölye bulgusu / NFF)

Tüm parametreler varsayımdır ve Config içinde ayarlanabilir. Gerçek ürün verisi kullanılmaz.

Kullanım:
  python fleet_sim.py --aircraft 50 --years 5 --seed 42 --start 2021-10-19 --out data_50_5y
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ----------------------------------------------------------------------------
# Sabitler
# ----------------------------------------------------------------------------
MODES = ["WEAR", "JAM", "MOTOR", "SENSOR"]          # aktüatör gizli arıza modları
WEAR, JAM, MOTOR, SENSOR = range(4)
ECU_MODES = ["NONE", "POWER", "COMM"]               # kontrol ünitesi gizli arıza modları
CHANNELS = ["TRK", "PRV"]
COMMANDS = np.array(["EXTEND", "RETRACT", "PRESET"])

# Kritik arıza kodları (dışarıya / eğlence sistemine giden L1 katmanı) — genel, uydurma numaralar
L1_ACT_FAULT = {0: 101, 1: 102}    # kanal -> kod
L1_ACT_JAM = {0: 111, 1: 112}
L1_ECU_FAULT = 201
L1_COMM_FAULT = 301


@dataclass
class Config:
    n_aircraft: int = 50
    seats_min: int = 28
    seats_max: int = 36                 # ortalama ~32
    start_date: str = "2024-01-01"
    years: float = 2.0
    seed: int = 42

    # --- filo / uçuş ---
    p_longhaul: float = 0.6
    lh_flights_mean: float = 1.1
    lh_dur: tuple = (8.0, 13.0)
    mh_flights_mean: float = 2.0
    mh_dur: tuple = (3.0, 6.0)
    p_ground_day: float = 0.02
    check_interval: tuple = (30, 60)    # gün, bakım ziyareti (IBITE burada yapılır)
    load_factor_ab: tuple = (8.0, 2.0)  # Beta dağılımı, ort. ~0.8

    # --- kullanım ---
    trk_moves_base: float = 3.0
    trk_moves_per_hour: float = 0.6
    prv_moves_per_flight: float = 1.2
    nominal_time_s: tuple = (12.0, 6.0) # tam strok süresi (TRK, PRV)

    # --- aktüatör ömrü (hareket sayısı cinsinden), mod başına Weibull ---
    mode_probs: tuple = (0.40, 0.25, 0.15, 0.20)
    weibull_shape: tuple = (3.0, 2.0, 2.5, 1.8)
    weibull_scale_trk: tuple = (22000.0, 35000.0, 30000.0, 40000.0)
    prv_scale_factor: float = 0.22      # PRV daha az hareket eder, ömür ölçeği orantılı
    degr_exponent: tuple = (3.0, 4.0, 3.0, 2.5)  # D = (c/L)^q ; q büyükse belirti geç görünür
    initial_age_max_trk: float = 15000.0       # başlangıçta takılı parçaların geçmiş kullanımı

    # --- olay olasılıkları (hareket başına) ---
    p_pax_jam: float = 0.0012           # yolcu kaynaklı sıkışma (arıza DEĞİL)
    p_base_timeout: float = 0.0003
    oc_coef: tuple = (0.004, 0.03, 0.006, 0.0)    # aşırı akım katsayısı (mod başına)
    oc_pow: tuple = (4, 5, 4, 1)
    to_coef: tuple = (0.010, 0.003, 0.002, 0.004) # zaman aşımı
    to_pow: tuple = (4, 4, 3, 3)
    p_pos_coef: float = 0.02            # konum uyuşmazlığı (sensör modu)
    p_ot_coef: float = 0.01             # motor aşırı ısınma (motor modu)
    current_gain: float = 35.0          # D=1'de ortalama akım artışı (limitin %'si)
    base_current: tuple = (55.0, 5.0)   # ortalama, std (limitin %'si)

    # --- kontrol ünitesi modları (yıllık başlama olasılığı) ---
    ecu_power_rate_yr: float = 0.025
    ecu_comm_rate_yr: float = 0.03
    ecu_power_fail_days: float = 45.0
    ecu_comm_removal_days: float = 30.0

    # --- bakım ---
    repair_p_per_day: float = 0.5       # arızadan sonra her gün onarılma olasılığı
    p_complaint_removal: float = 0.008  # L1 sıkışma kodu görülen gün şikâyete bağlı söküm olasılığı
    l1_jam_threshold: int = 1           # günde bu kadar aşırı akım olayı -> kritik sıkışma kodu
    nff_threshold_D: float = 0.6        # sökülen parçada D bunun altındaysa atölye NFF bulur
    p_power_confirmed: float = 0.6
    p_comm_confirmed: float = 0.25

    # --- veri kalitesi ---
    p_missing_download: float = 0.02    # günlük sayaç/ECU verisinin inmemesi
    l1_poll_min: int = 5                # kritik kodların sorgulama aralığı (dk)

    flush_every_days: int = 30


# ----------------------------------------------------------------------------
# Yardımcılar
# ----------------------------------------------------------------------------
class TableWriter:
    """Tabloları parça parça Parquet'e yazar (500 uçakta bellek taşmasın)."""

    def __init__(self, out: Path):
        self.out = out
        self.buf: dict[str, list[pd.DataFrame]] = {}
        self.writers: dict[str, pq.ParquetWriter] = {}

    def add(self, name: str, df: pd.DataFrame):
        if df is not None and len(df):
            self.buf.setdefault(name, []).append(df)

    def flush(self):
        for name, parts in self.buf.items():
            if not parts:
                continue
            df = pd.concat(parts, ignore_index=True)
            table = pa.Table.from_pandas(df, preserve_index=False)
            if name not in self.writers:
                self.writers[name] = pq.ParquetWriter(self.out / f"{name}.parquet", table.schema)
            self.writers[name].write_table(table.cast(self.writers[name].schema))
            self.buf[name] = []

    def close(self):
        self.flush()
        for w in self.writers.values():
            w.close()


def weibull_conditional(rng, shape, scale, age):
    """Yaşı `age` olan bir parçanın toplam ömrü (koşullu Weibull)."""
    u = rng.random(np.shape(age))
    return scale * ((age / scale) ** shape - np.log(u)) ** (1.0 / shape)


# ----------------------------------------------------------------------------
# Simülatör
# ----------------------------------------------------------------------------
class FleetSimulator:
    def __init__(self, cfg: Config, out: Path):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.w = TableWriter(out)
        self.next_serial = 1
        self.next_ecu_serial = 1
        self._build_fleet()

    # ---------------- filo kurulumu ----------------
    def _build_fleet(self):
        c, r = self.cfg, self.rng
        A = c.n_aircraft
        lh = r.random(A) < c.p_longhaul
        self.ac_id = np.array([f"AC{i+1:03d}" for i in range(A)])
        self.ac_lh = lh
        self.ac_flights_mean = np.where(lh, c.lh_flights_mean, c.mh_flights_mean)
        self.ac_dur_lo = np.where(lh, c.lh_dur[0], c.mh_dur[0])
        self.ac_dur_hi = np.where(lh, c.lh_dur[1], c.mh_dur[1])
        self.ac_check_int = r.integers(c.check_interval[0], c.check_interval[1] + 1, A)
        self.ac_next_check = r.integers(0, self.ac_check_int)
        n_seats = r.integers(c.seats_min, c.seats_max + 1, A)

        seat_rows, letters = [], ["A", "D", "G", "K"]
        for a in range(A):
            for s in range(n_seats[a]):
                row, col = divmod(s, 4)
                seat_rows.append((self.ac_id[a], a, f"{self.ac_id[a]}-{row+1:02d}{letters[col]}",
                                  row + 1, "WINDOW" if col in (0, 3) else "CENTER"))
        seats = pd.DataFrame(seat_rows, columns=["aircraft_id", "ac_idx", "seat_id", "row", "seat_pos"])
        self.S = len(seats)
        self.seat_ac = seats["ac_idx"].to_numpy()
        self.seat_id = seats["seat_id"].to_numpy()

        # kanal dizileri: indeks k = 2*s + ch
        self.N = 2 * self.S
        self.act_seat = np.repeat(np.arange(self.S), 2)
        self.act_ch = np.tile([0, 1], self.S)
        self.act_serial = np.zeros(self.N, dtype=np.int64)
        self.act_mode = np.zeros(self.N, dtype=np.int8)
        self.act_life = np.zeros(self.N)
        self.act_moves = np.zeros(self.N)            # gerçek kullanım (takıldığından beri)
        self.act_base_cur = np.zeros(self.N)
        self.act_jam_pos = np.zeros(self.N)
        self.act_install_day = np.zeros(self.N, dtype=np.int64)
        self.cnt_starts = np.zeros(self.N)           # ECU sayacı (ECU veya aktüatör değişince sıfırlanır)
        self.cnt_runtime = np.zeros(self.N)
        self.act_failed = np.zeros(self.N, dtype=bool)
        self.act_fail_day = np.full(self.N, -1)
        self.l1_jam_active = np.zeros(self.N, dtype=bool)

        self.installs = []   # aktüatör takma kayıtları (bakım sisteminden bilinir)
        self.truth = []      # gizli gerçek: seri -> mod, ömür, arıza günü

        initial_age = r.random(self.N) * c.initial_age_max_trk * np.where(self.act_ch == 0, 1.0, c.prv_scale_factor)
        self._new_actuators(np.arange(self.N), day=0, age=initial_age)

        # kontrol üniteleri
        self.ecu_serial = np.zeros(self.S, dtype=np.int64)
        self.ecu_mode = np.zeros(self.S, dtype=np.int8)
        self.ecu_onset = np.full(self.S, -1)
        self.ecu_event_day = np.full(self.S, -1)     # POWER: arıza günü, COMM: söküm günü
        self.ecu_failed = np.zeros(self.S, dtype=bool)
        self.ecu_starts_cum = np.zeros(self.S)
        self.ecu_hours_cum = np.zeros(self.S)
        self.ecu_installs = []
        self._new_ecus(np.arange(self.S), day=0)

        fleet = pd.DataFrame({
            "aircraft_id": self.ac_id,
            "profile": np.where(lh, "LONG_HAUL", "MEDIUM_HAUL"),
            "n_seats": n_seats,
            "check_interval_days": self.ac_check_int,
        })
        fleet.to_parquet(self.out / "fleet.parquet", index=False)
        seats.drop(columns="ac_idx").to_parquet(self.out / "seats.parquet", index=False)

    def _new_actuators(self, idx, day, age=None):
        c, r = self.cfg, self.rng
        n = len(idx)
        if n == 0:
            return
        if age is None:
            age = np.zeros(n)
        mode = r.choice(4, size=n, p=c.mode_probs)
        shape = np.array(c.weibull_shape)[mode]
        scale = np.array(c.weibull_scale_trk)[mode] * np.where(self.act_ch[idx] == 0, 1.0, c.prv_scale_factor)
        life = weibull_conditional(r, shape, scale, age)
        serials = np.arange(self.next_serial, self.next_serial + n)
        self.next_serial += n
        self.act_serial[idx] = serials
        self.act_mode[idx] = mode
        self.act_life[idx] = life
        self.act_moves[idx] = age
        self.act_base_cur[idx] = r.normal(*c.base_current, n)
        self.act_jam_pos[idx] = r.uniform(15, 85, n)
        self.act_install_day[idx] = day
        self.cnt_starts[idx] = 0
        self.cnt_runtime[idx] = 0
        self.act_failed[idx] = False
        self.act_fail_day[idx] = -1
        self.l1_jam_active[idx] = False
        for k, s in zip(idx, serials):
            self.installs.append((int(s), self.seat_id[self.act_seat[k]], CHANNELS[self.act_ch[k]], int(day)))
            self.truth.append({"actuator_serial": int(s), "seat_id": self.seat_id[self.act_seat[k]],
                               "channel": CHANNELS[self.act_ch[k]], "install_day": int(day),
                               "hidden_mode": MODES[self.act_mode[k]], "hidden_life_moves": float(self.act_life[k]),
                               "moves_at_install": float(self.act_moves[k]),
                               "failure_day": -1, "removal_day": -1, "removal_reason": None})

    def _new_ecus(self, sidx, day):
        n = len(sidx)
        if n == 0:
            return
        ser = np.arange(self.next_ecu_serial, self.next_ecu_serial + n)
        self.next_ecu_serial += n
        self.ecu_serial[sidx] = ser
        self.ecu_mode[sidx] = 0
        self.ecu_onset[sidx] = -1
        self.ecu_event_day[sidx] = -1
        self.ecu_failed[sidx] = False
        self.ecu_starts_cum[sidx] = 0
        self.ecu_hours_cum[sidx] = 0
        # yeni ECU: o koltuğun kanal sayaçları sıfırlanır
        k = np.concatenate([2 * sidx, 2 * sidx + 1])
        self.cnt_starts[k] = 0
        self.cnt_runtime[k] = 0
        for s, e in zip(sidx, ser):
            self.ecu_installs.append((int(e), self.seat_id[s], int(day)))

    # ---------------- yardımcı: olay satırları ----------------
    def _events(self, ts_day, k_idx, counts, log_type, code, cur, pos, temp, cmd=None):
        """k_idx aktüatörleri için counts kadar olay satırı üretir."""
        rep = np.repeat(k_idx, counts)
        if len(rep) == 0:
            return None
        r = self.rng
        n = len(rep)
        secs = r.integers(6 * 3600, 23 * 3600, n)  # olaylar gün içinde
        s = self.act_seat[rep]
        return pd.DataFrame({
            "timestamp": ts_day + pd.to_timedelta(secs, unit="s"),
            "aircraft_id": self.ac_id[self.seat_ac[s]],
            "seat_id": self.seat_id[s],
            "ecu_serial": self.ecu_serial[s],
            "channel": np.array(CHANNELS)[self.act_ch[rep]],
            "log_type": log_type,
            "event_code": code,
            "status": "SET",
            "ctx_current_pct": np.round(cur(rep, n), 1) if cur else np.nan,
            "ctx_position_pct": np.round(np.clip(pos(rep, n), 0, 100), 1) if pos else np.nan,
            "ctx_command": cmd(rep, n) if cmd else COMMANDS[r.integers(0, 3, n)],
            "ctx_temp_c": np.round(temp(rep, n), 1) if temp else np.nan,
        })

    def _l1(self, ts_day, seat_idx, code, status, secs=None):
        if len(seat_idx) == 0:
            return None
        r = self.rng
        n = len(seat_idx)
        if secs is None:
            secs = r.integers(6 * 3600, 23 * 3600, n)
        poll = self.cfg.l1_poll_min * 60
        secs = (np.ceil(secs / poll) * poll).astype(int)    # sorgulama aralığına yuvarlama
        return pd.DataFrame({
            "timestamp": ts_day + pd.to_timedelta(secs, unit="s"),
            "aircraft_id": self.ac_id[self.seat_ac[seat_idx]],
            "seat_id": self.seat_id[seat_idx],
            "fault_code": np.broadcast_to(code, n),
            "state": status,
        })

    def _maint(self, day_ts, seat_idx, component, serials, reason, finding):
        if len(seat_idx) == 0:
            return
        self.w.add("maintenance", pd.DataFrame({
            "date": day_ts,
            "aircraft_id": self.ac_id[self.seat_ac[seat_idx]],
            "seat_id": self.seat_id[seat_idx],
            "component": component,
            "removed_serial": serials,
            "removal_reason": reason,
            "shop_finding": finding,
        }))

    # ---------------- ana döngü ----------------
    def run(self):
        c, r = self.cfg, self.rng
        n_days = int(round(365 * c.years))
        start = pd.Timestamp(c.start_date)
        A, S, N = c.n_aircraft, self.S, self.N
        ch0 = self.act_ch == 0
        t0 = time.time()

        for d in range(n_days):
            day_ts = start + pd.Timedelta(days=d)

            # ---- 1. uçuş planı ----
            check_today = (d == self.ac_next_check)
            self.ac_next_check = np.where(check_today, d + self.ac_check_int, self.ac_next_check)
            ground = check_today | (r.random(A) < c.p_ground_day)
            nfl = r.poisson(self.ac_flights_mean)
            nfl = np.where(self.ac_lh, np.minimum(nfl, 3), np.minimum(nfl, 4))
            nfl = np.where(ground, 0, nfl)
            dur = r.uniform(self.ac_dur_lo, self.ac_dur_hi)
            lf = r.beta(*c.load_factor_ab, A)
            self.w.add("flights_daily", pd.DataFrame({
                "date": day_ts, "aircraft_id": self.ac_id, "n_flights": nfl,
                "avg_block_hours": np.round(np.where(nfl > 0, dur, 0), 2),
                "load_factor": np.round(np.where(nfl > 0, lf, 0), 3),
                "maintenance_check": check_today}))

            # ---- 2. kullanım ----
            sa = self.seat_ac
            seat_inop = self.ecu_failed | self.act_failed[0::2] | self.act_failed[1::2]
            occ = r.binomial(nfl[sa], lf[sa])
            occ = np.where(seat_inop, 0, occ)
            trk_m = r.poisson(occ * (c.trk_moves_base + c.trk_moves_per_hour * dur[sa]))
            prv_m = r.poisson(occ * c.prv_moves_per_flight)
            moves = np.empty(N)
            moves[0::2], moves[1::2] = trk_m, prv_m
            moves[self.act_failed] = 0
            self.act_moves += moves

            # ---- 3. gizli bozulma ----
            q = np.array(c.degr_exponent)[self.act_mode]
            D = np.clip(self.act_moves / self.act_life, 0, None) ** q
            Dc = np.minimum(D, 1.2)
            mode = self.act_mode
            cur_mean = self.act_base_cur + c.current_gain * Dc * np.where(mode == SENSOR, 0.2, 1.0)

            # ---- 4. sayaçlar ----
            nom = np.where(ch0, c.nominal_time_s[0], c.nominal_time_s[1])
            slow = 1 + 0.25 * Dc * np.isin(mode, [WEAR, JAM]) + 0.10 * Dc * (mode == MOTOR)
            stroke_frac = np.clip(r.normal(0.65, 0.12, N), 0.3, 1.0)
            runtime = moves * nom * stroke_frac * slow
            self.cnt_starts += moves
            self.cnt_runtime += runtime

            # ---- 5. olay logu (L2) ----
            evs = []
            idx_all = np.arange(N)
            # 5a. aşırı akım: yolcu kaynaklı + bozulma kaynaklı (aynı kod, farklı bağlam!)
            n_pax = r.poisson(moves * c.p_pax_jam)
            oc_p = np.array(c.oc_coef)[mode] * Dc ** np.array(c.oc_pow)[mode]
            n_deg = r.poisson(moves * oc_p)
            e = self._events(day_ts, idx_all, n_pax, "ERROR", "OVERCURRENT",
                             cur=lambda k, n: r.uniform(100, 115, n),
                             pos=lambda k, n: r.uniform(5, 95, n),
                             temp=lambda k, n: r.normal(42, 4, n))
            evs.append(e)

            def deg_pos(k, n):
                m = self.act_mode[k]
                p = np.where(m == JAM, r.normal(self.act_jam_pos[k], 3, n),
                    np.where(m == WEAR, np.where(r.random(n) < 0.5, r.uniform(0, 10, n), r.uniform(90, 100, n)),
                             r.uniform(0, 100, n)))
                return p
            e = self._events(day_ts, idx_all, n_deg, "ERROR", "OVERCURRENT",
                             cur=lambda k, n: 100 + np.abs(r.normal(np.where(self.act_mode[k] == MOTOR, 10, 6), 4, n)),
                             pos=deg_pos,
                             temp=lambda k, n: r.normal(45, 5, n) + 15 * np.minimum(D[k], 1.2) * (self.act_mode[k] == MOTOR))
            evs.append(e)
            n_oc = n_pax + n_deg

            # 5b. zaman aşımı
            to_p = c.p_base_timeout + np.array(c.to_coef)[mode] * Dc ** np.array(c.to_pow)[mode]
            n_to = r.poisson(moves * to_p)
            evs.append(self._events(day_ts, idx_all, n_to, "ERROR", "ACTUATOR_TIMEOUT",
                                    cur=lambda k, n: cur_mean[k] + r.normal(0, 6, n),
                                    pos=lambda k, n: r.uniform(20, 80, n),
                                    temp=lambda k, n: r.normal(43, 4, n)))
            # 5c. konum uyuşmazlığı (sensör kayması)
            n_pos = r.poisson(moves * c.p_pos_coef * Dc ** 3 * (mode == SENSOR))
            evs.append(self._events(day_ts, idx_all, n_pos, "ERROR", "POSITION_MISMATCH",
                                    cur=lambda k, n: cur_mean[k] + r.normal(0, 5, n),
                                    pos=lambda k, n: r.uniform(0, 100, n),
                                    temp=lambda k, n: r.normal(42, 4, n)))
            # 5d. motor aşırı ısınma
            n_ot = r.poisson(moves * c.p_ot_coef * Dc ** 4 * (mode == MOTOR))
            evs.append(self._events(day_ts, idx_all, n_ot, "ERROR", "MOTOR_OVERTEMP",
                                    cur=lambda k, n: cur_mean[k] + r.normal(5, 5, n),
                                    pos=lambda k, n: r.uniform(0, 100, n),
                                    temp=lambda k, n: r.normal(78, 4, n)))

            # ---- 6. kontrol ünitesi modları ----
            daily_hazard_p = c.ecu_power_rate_yr / 365
            daily_hazard_c = c.ecu_comm_rate_yr / 365
            healthy = (self.ecu_mode == 0)
            new_p = healthy & (r.random(S) < daily_hazard_p)
            new_c = healthy & ~new_p & (r.random(S) < daily_hazard_c)
            self.ecu_mode[new_p], self.ecu_mode[new_c] = 1, 2
            self.ecu_onset[new_p | new_c] = d
            self.ecu_event_day[new_p] = d + np.maximum(1, r.exponential(c.ecu_power_fail_days, new_p.sum())).astype(int)
            self.ecu_event_day[new_c] = d + np.maximum(1, r.exponential(c.ecu_comm_removal_days, new_c.sum())).astype(int)

            powered = nfl[sa] > 0
            age_on = np.where(self.ecu_onset >= 0, d - self.ecu_onset, 0)
            # güç sorunu: beklenmeyen yeniden başlatmalar ve düşük gerilim
            pw = (self.ecu_mode == 1) & ~self.ecu_failed & powered
            lam_reset = np.where(pw, 0.08 * np.exp(np.minimum(age_on, 120) / 20.0), 0)
            n_reset = r.poisson(np.minimum(lam_reset, 8))
            n_uv = r.poisson(lam_reset * 0.7)
            # haberleşme sorunu: eğlence sistemi bağlantı kaybı
            cm = (self.ecu_mode == 2) & ~self.ecu_failed & powered
            lam_link = np.where(cm, 0.3 + 0.03 * np.minimum(age_on, 90), 0)
            n_link = r.poisson(lam_link)
            seat_rows = np.arange(S)
            for cnt, code in ((n_reset, "UNEXPECTED_RESET"), (n_uv, "SUPPLY_UNDERVOLTAGE"), (n_link, "IFE_LINK_LOST")):
                rep = np.repeat(seat_rows, cnt)
                if len(rep):
                    evs.append(pd.DataFrame({
                        "timestamp": day_ts + pd.to_timedelta(r.integers(6 * 3600, 23 * 3600, len(rep)), unit="s"),
                        "aircraft_id": self.ac_id[self.seat_ac[rep]], "seat_id": self.seat_id[rep],
                        "ecu_serial": self.ecu_serial[rep], "channel": "ECU",
                        "log_type": "ERROR" if code != "IFE_LINK_LOST" else "EVENT",
                        "event_code": code, "status": "SET",
                        "ctx_current_pct": np.nan, "ctx_position_pct": np.nan, "ctx_command": None,
                        "ctx_temp_c": np.round(r.normal(42, 4, len(rep)), 1)}))
            ev_df = [x for x in evs if x is not None and len(x)]
            if ev_df:
                self.w.add("event_log", pd.concat(ev_df, ignore_index=True))

            # ---- 7. kritik arıza kodları (L1) ----
            l1 = []
            # aktüatör sıkışma kodu: gün içinde eşik kadar aşırı akım
            jam_now = n_oc >= c.l1_jam_threshold
            set_k = np.where(jam_now & ~self.l1_jam_active)[0]
            clr_k = np.where(~jam_now & self.l1_jam_active & (moves > 0))[0]
            for ks, st in ((set_k, "SET"), (clr_k, "CLEAR")):
                for ch in (0, 1):
                    kk = ks[self.act_ch[ks] == ch]
                    l1.append(self._l1(day_ts, self.act_seat[kk], L1_ACT_JAM[ch], st))
            self.l1_jam_active[set_k] = True
            self.l1_jam_active[clr_k] = False
            # ECU kodları
            l1.append(self._l1(day_ts, np.where(n_reset >= 3)[0], L1_ECU_FAULT, "SET"))
            l1.append(self._l1(day_ts, np.where(n_link >= 2)[0], L1_COMM_FAULT, "SET"))

            # ---- 8. fonksiyonel arızalar ----
            new_fail = (D >= 1.0) & ~self.act_failed
            fk = np.where(new_fail)[0]
            self.act_failed[fk] = True
            self.act_fail_day[fk] = d
            for ch in (0, 1):
                kk = fk[self.act_ch[fk] == ch]
                l1.append(self._l1(day_ts, self.act_seat[kk], L1_ACT_FAULT[ch], "SET"))
            for k in fk:
                self.truth[self.act_serial[k] - 1]["failure_day"] = d
            ev = self._events(day_ts, fk, np.ones(len(fk), dtype=int), "ERROR", "ACTUATOR_FAILURE",
                              cur=lambda k, n: 100 + np.abs(r.normal(10, 5, n)),
                              pos=lambda k, n: r.uniform(0, 100, n),
                              temp=lambda k, n: r.normal(48, 5, n))
            if ev is not None:
                self.w.add("event_log", ev)

            ecu_fail_now = (self.ecu_mode == 1) & ~self.ecu_failed & (self.ecu_event_day == d)
            self.ecu_failed[ecu_fail_now] = True
            l1.append(self._l1(day_ts, np.where(ecu_fail_now)[0], L1_ECU_FAULT, "SET"))
            l1 = [x for x in l1 if x is not None and len(x)]
            if l1:
                self.w.add("fault_codes", pd.concat(l1, ignore_index=True))

            # ---- 9. bakım ----
            # 9a. arızalı aktüatör değişimi
            rep_k = np.where(self.act_failed & (self.act_fail_day < d) & (r.random(N) < c.repair_p_per_day))[0]
            if len(rep_k):
                self._maint(day_ts, self.act_seat[rep_k], np.array(CHANNELS)[self.act_ch[rep_k]],
                            self.act_serial[rep_k], "FUNCTIONAL_FAILURE",
                            np.array([f"CONFIRMED_{MODES[m]}" for m in self.act_mode[rep_k]]))
                for k in rep_k:
                    t = self.truth[self.act_serial[k] - 1]
                    t["removal_day"], t["removal_reason"] = d, "FUNCTIONAL_FAILURE"
                for ch in (0, 1):
                    kk = rep_k[self.act_ch[rep_k] == ch]
                    self.w.add("fault_codes", self._l1(day_ts, self.act_seat[kk], L1_ACT_FAULT[ch], "CLEAR"))
                self._new_actuators(rep_k, d)
            # 9b. şikâyete bağlı erken söküm (sıkışma kodu görüldü, parça henüz arızalı değil)
            cand = np.where(jam_now & ~self.act_failed & (r.random(N) < c.p_complaint_removal))[0]
            cand = np.setdiff1d(cand, rep_k)
            if len(cand):
                nff = D[cand] < c.nff_threshold_D
                finding = np.where(nff, "NFF", np.array([f"CONFIRMED_{MODES[m]}" for m in self.act_mode[cand]]))
                self._maint(day_ts, self.act_seat[cand], np.array(CHANNELS)[self.act_ch[cand]],
                            self.act_serial[cand], "JAM_COMPLAINT", finding)
                for k in cand:
                    t = self.truth[self.act_serial[k] - 1]
                    t["removal_day"], t["removal_reason"] = d, "JAM_COMPLAINT"
                self._new_actuators(cand, d)
            # 9c. kontrol ünitesi değişimi
            ecu_rep = np.where((self.ecu_failed & (r.random(S) < c.repair_p_per_day)) |
                               ((self.ecu_mode == 2) & (self.ecu_event_day == d)))[0]
            if len(ecu_rep):
                m = self.ecu_mode[ecu_rep]
                conf = np.where(m == 1, r.random(len(ecu_rep)) < c.p_power_confirmed,
                                r.random(len(ecu_rep)) < c.p_comm_confirmed)
                finding = np.where(conf, np.where(m == 1, "CONFIRMED_POWER", "CONFIRMED_COMM"), "NFF")
                reason = np.where(m == 1, "FUNCTIONAL_FAILURE", "COMM_COMPLAINT")
                self._maint(day_ts, ecu_rep, "ECU", self.ecu_serial[ecu_rep], reason, finding)
                self.ecu_truth_rows(ecu_rep, d)
                self._new_ecus(ecu_rep, d)

            # ---- 10. IBITE (bakım ziyaretinde) ----
            ck_seats = np.where(check_today[sa] & ~(self.ecu_failed))[0]
            if len(ck_seats):
                kk = np.concatenate([2 * ck_seats, 2 * ck_seats + 1])
                kk = kk[~self.act_failed[kk]]
                nomk = nom[kk]
                travel = nomk * (1 + 0.25 * Dc[kk] * np.isin(mode[kk], [WEAR, JAM])
                                 + 0.08 * Dc[kk] * (mode[kk] == MOTOR)) * r.normal(1, 0.03, len(kk))
                peak = cur_mean[kk] + 12 + r.normal(0, 3, len(kk)) + 10 * Dc[kk] * (mode[kk] == JAM)
                calib = r.normal(0, 0.6, len(kk)) + 8 * Dc[kk] * (mode[kk] == SENSOR)
                fail = (peak > 100) | (np.abs(calib) > 5) | (travel > 1.3 * nomk)
                self.w.add("ibite", pd.DataFrame({
                    "date": day_ts, "aircraft_id": self.ac_id[self.seat_ac[self.act_seat[kk]]],
                    "seat_id": self.seat_id[self.act_seat[kk]], "ecu_serial": self.ecu_serial[self.act_seat[kk]],
                    "channel": np.array(CHANNELS)[self.act_ch[kk]],
                    "travel_time_s": np.round(travel, 2), "peak_current_pct": np.round(peak, 1),
                    "calib_offset_pct": np.round(calib, 2), "result": np.where(fail, "FAIL", "PASS")}))

            # ---- 11. günlük sayaç ve ECU özeti (indirme kaybı ile) ----
            ecu_starts = np.where(powered, nfl[sa] + 1, 0) + n_reset
            self.ecu_starts_cum += ecu_starts
            on_hours = np.where(powered, nfl[sa] * (dur[sa] + 1.0), 0)
            self.ecu_hours_cum += on_hours
            mot = (mode == MOTOR) * Dc
            maxt = np.maximum.reduceat(np.where(moves > 0, 15 * mot, 0), np.arange(0, N, 2))
            max_temp = np.where(powered, r.normal(40, 3, S) + 5 * self.ac_lh[sa] + maxt, np.nan)
            got = r.random(S) >= c.p_missing_download
            gs = np.where(got)[0]
            self.w.add("ecu_daily", pd.DataFrame({
                "date": day_ts, "aircraft_id": self.ac_id[sa[gs]], "seat_id": self.seat_id[gs],
                "ecu_serial": self.ecu_serial[gs], "starts_day": ecu_starts[gs],
                "starts_cum": self.ecu_starts_cum[gs], "on_hours_day": np.round(on_hours[gs], 2),
                "on_hours_cum": np.round(self.ecu_hours_cum[gs], 1), "max_temp_c": np.round(max_temp[gs], 1)}))
            gk = np.where(np.repeat(got, 2))[0]
            self.w.add("counters_daily", pd.DataFrame({
                "date": day_ts, "aircraft_id": self.ac_id[sa[self.act_seat[gk]]],
                "seat_id": self.seat_id[self.act_seat[gk]], "ecu_serial": self.ecu_serial[self.act_seat[gk]],
                "channel": np.array(CHANNELS)[self.act_ch[gk]],
                "starts_day": moves[gk].astype(np.int32), "starts_cum": self.cnt_starts[gk].astype(np.int64),
                "cycles_cum": (self.cnt_starts[gk] // 2).astype(np.int64),
                "runtime_s_day": np.round(runtime[gk], 1), "runtime_s_cum": np.round(self.cnt_runtime[gk], 0)}))

            # gizli günlük sağlık (sadece değerlendirme için)
            self.w.add("truth_daily", pd.DataFrame({
                "date": day_ts, "actuator_serial": self.act_serial, "seat_id": self.seat_id[self.act_seat],
                "channel": np.array(CHANNELS)[self.act_ch], "health_D": np.round(np.minimum(D, 1.0), 4).astype(np.float32),
                "failed": self.act_failed}))

            if (d + 1) % c.flush_every_days == 0:
                self.w.flush()
                print(f"  gün {d+1}/{n_days}  ({time.time()-t0:.0f} sn)", flush=True)

        self.w.close()
        self._write_static(n_days)

    def ecu_truth_rows(self, sidx, d):
        if not hasattr(self, "ecu_truth"):
            self.ecu_truth = []
        for s in sidx:
            self.ecu_truth.append({"ecu_serial": int(self.ecu_serial[s]), "seat_id": self.seat_id[s],
                                   "hidden_mode": ECU_MODES[self.ecu_mode[s]], "onset_day": int(self.ecu_onset[s]),
                                   "removal_day": int(d)})

    def _write_static(self, n_days):
        start = pd.Timestamp(self.cfg.start_date)
        inst = pd.DataFrame(self.installs, columns=["actuator_serial", "seat_id", "channel", "install_day"])
        inst["install_date"] = start + pd.to_timedelta(inst["install_day"], unit="D")
        inst.drop(columns="install_day").to_parquet(self.out / "actuator_installs.parquet", index=False)
        einst = pd.DataFrame(self.ecu_installs, columns=["ecu_serial", "seat_id", "install_day"])
        einst["install_date"] = start + pd.to_timedelta(einst["install_day"], unit="D")
        einst.drop(columns="install_day").to_parquet(self.out / "ecu_installs.parquet", index=False)

        tr = pd.DataFrame(self.truth)
        for col in ("install_day", "failure_day", "removal_day"):
            tr[col.replace("_day", "_date")] = np.where(tr[col] >= 0, start + pd.to_timedelta(tr[col], unit="D"), pd.NaT)
        tr.drop(columns=["install_day", "failure_day", "removal_day"]).to_parquet(
            self.out / "truth_actuators.parquet", index=False)
        et = pd.DataFrame(getattr(self, "ecu_truth", []),
                          columns=["ecu_serial", "seat_id", "hidden_mode", "onset_day", "removal_day"])
        et.to_parquet(self.out / "truth_ecus.parquet", index=False)
        cfg = asdict(self.cfg)
        cfg["n_seats_total"] = int(self.S)
        cfg["n_actuators_total"] = int(self.N)
        cfg["n_days"] = n_days
        (self.out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aircraft", type=int, default=50)
    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data_50")
    ap.add_argument("--start", type=str, default="2024-01-01")
    a = ap.parse_args()
    cfg = Config(n_aircraft=a.aircraft, years=a.years, seed=a.seed, start_date=a.start)
    print(f"Simülasyon: {cfg.n_aircraft} uçak, {cfg.years} yıl, seed={cfg.seed}")
    sim = FleetSimulator(cfg, Path(a.out))
    print(f"  {sim.S} koltuk, {sim.N} aktüatör")
    sim.run()
    print("Bitti ->", a.out)


if __name__ == "__main__":
    main()
