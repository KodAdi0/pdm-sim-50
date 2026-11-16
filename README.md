# PDM-SIM-50: Koltuk aktüatörü kestirimci bakımı için sentetik filo simülatörü

YTÜ MKT6107 Mekatronik Mühendisliği Uygulamalarında Yapay Zekâ dönem projesi (Ömer Ergün, Ali Narin).

Genel (belirli bir ürüne özgü olmayan) bir business class koltuk hareket sistemini modelleyen sentetik veri simülatörüdür. Her koltukta 1 kontrol ünitesi (ECU) ve 2 aktüatör bulunur: TRK (kaydırma) ve PRV (gizlilik paneli). Gerçek ürün ya da uçak verisi içermez; tüm parametreler varsayımdır.

## Veriyi yeniden üretme

```
pip install -r requirements.txt
python fleet_sim.py --aircraft 50 --years 5 --seed 42 --start 2021-10-19 --out data_50_5y
```

Aynı tohum aynı veriyi üretir. Harici test filoları için `--seed 43` ve `--seed 44` kullanılır.

## İçerik

- `fleet_sim.py`: simülatör (v1.0)
- `config.json`: makaledeki PDM-SIM-50 v1.0 verisinin üretildiği parametrelerin kaydı (simülatör bu dosyayı çıktı klasörüne kendisi yazar)
- `truth_*` tabloları yalnız değerlendirme içindir; modellerde öznitelik olarak kullanılmaz.

## Bilinen sorun (v1.0)

`truth_daily` tablosu, değişim gününde yeni seri numarasına sökülen parçanın `health_D` değerini yazar. v1.1'de düzeltilecektir.

## Lisans

MIT (bkz. `LICENSE`).

## Üretken yapay zekâ beyanı

Simülatör kodu Claude (Anthropic) yardımıyla taslaklanmış ve yazarlarca gözden geçirilmiştir.
