# waifu2x-hsa

waifu2xをHSAで動かす予定！

## TODO

* [x] Rustで取り敢えず動く版
* [ ] Rust高速版 (シングルスレッド)
* [ ] Rust高速版 (マルチスレッド)
* [ ] HSAで取り敢えず動く版
* [ ] HSA高速版

## Performance

||Test environments|
|:--|:---------------------------------------------|
|Source Image|https://github.com/nagadomi/waifu2x/blob/master/images/miku_small.png|
|Method|scale2.0x|
|Model|https://github.com/nagadomi/waifu2x/tree/master/models/anime_style_art|
|CPU(Haswell)|Intel Xeon E5-1650v3 (3.50GHz, 6-Core, 12-SMT)|
|GPU(Tonga)|AMD Radeon R9 285|

|impl|type|threads|time[ms]|GFLOPS|
|:---|:---|:------|:-------|:-----|
|03da5c565067765fdada59620769efb1d81955d5|Haswell|1|38400|2.1|
|7c8f1065a63bf93f1e83ebadb7865f06501f3039|Haswell|1|13693|6.1|
|waifu2x-converter-cpp|Haswell|1|2314|38|
|waifu2x-converter-cpp|Haswell|6|548|163|
|waifu2x-converter-cpp|Haswell|12|371|239|
|waifu2x-converter-cpp|Tonga|-|131|678|
|waifu2x-opt|Haswell|1|26276||
|waifu2x-opt|Haswell|6|6693||
|waifu2x-opt|Haswell|12|4957||
|waifu2x.py|Haswell|1|139740||
