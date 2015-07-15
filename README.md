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
|CPU|Intel Xeon E5-1650v3 (3.50GHz, 6-Core, 12-SMT)|
|Source Image|https://github.com/nagadomi/waifu2x/blob/master/images/miku_small.png|
|Method|scale2.0x|
|Model|https://github.com/nagadomi/waifu2x/tree/master/models/anime_style_art|

|impl|threads|time[ms]|
|:---|:------|:-------|
|03da5c565067765fdada59620769efb1d81955d5|1|38400|
|waifu2x-opt|1|26276|
|waifu2x-opt|6|6693|
|waifu2x-opt|12|4957|
