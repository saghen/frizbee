## [0.12.0](https://github.com/saghen/frizbee/compare/v0.11.0..v0.12.0) - 2026-07-30

### Breaking Changes

- per-pattern config (`PatternConfig`) ([6d2f478](https://github.com/saghen/frizbee/commit/6d2f4789aba0e9c74865349415963914ebc68da8))

### Features

- add `scoring.max_needle_len()` ([8f14c04](https://github.com/saghen/frizbee/commit/8f14c043d379daab64c86a8717511817cb551b41))

### Bug Fixes

- ensure delimiter bonus applied on first char ([da1adf3](https://github.com/saghen/frizbee/commit/da1adf33c6907c62604d0bd8e600593d19bdb6c3))
- remaining arithmetic panics ([3476999](https://github.com/saghen/frizbee/commit/3476999a9d5aaa0f8c9534bfba22adbac57d266c))
- ignored unicode matching indices ([074252f](https://github.com/saghen/frizbee/commit/074252fe7c3999c128ac453d37f4a38917c8e98e))
- pattern space double escape ([940bcb9](https://github.com/saghen/frizbee/commit/940bcb981a70695f0ec215cc535ea6ac3d75c78f))
- unicode match end col ([b33d28c](https://github.com/saghen/frizbee/commit/b33d28ce0e0ced873b4de69c1f041e2371d098e8))
- mixed multi-byte unicode failing prefilter ([b16ad8c](https://github.com/saghen/frizbee/commit/b16ad8c87d3fa05b25f44c91f0ae4852832c6a6f))
- check all instruction sets in sse smith waterman ([dd38a4b](https://github.com/saghen/frizbee/commit/dd38a4b517c999c81966f4a13ec97a1f6a945861))
- greedy gap penalty overflow and end col truncation ([4900c18](https://github.com/saghen/frizbee/commit/4900c189996d1ede58fb03a65ede9a5fa0b057ad))
- unsound u8 backend selection and overflow guard bounds ([927efbe](https://github.com/saghen/frizbee/commit/927efbea5f396a46d075b280e364a23ded4ee0ae))
- case flipped unicode match hidden by decoy last byte in prefilter ([bf4a247](https://github.com/saghen/frizbee/commit/bf4a247afb92f3ab8657a86b6d7cb7485d83da5b))
- match_list_indices dropping score 0 haystacks ([e4fd349](https://github.com/saghen/frizbee/commit/e4fd349052b01b14b78b3607ea0daf9c5cc144d1))
- guard against gap penalty overflowing u16/u8 ([250ed36](https://github.com/saghen/frizbee/commit/250ed364d5f2bb1d58567cc4b5c7ba789bb9378a))
- switch from castsi -> zextsi ([8da6dc9](https://github.com/saghen/frizbee/commit/8da6dc982d2d2bbadf9fbd751abc88f6d2aebd3d))
- unicode-aware char count for scoring guard ([48f683e](https://github.com/saghen/frizbee/commit/48f683ee86fd7353a017f3d1ab32b5fae2b24217))
- pattern escape parsing ([95efcf6](https://github.com/saghen/frizbee/commit/95efcf6ff232ead3e542b932696d10b089eef8b2))

### Refactor

- consolidate `include_exact` logic ([9b48b09](https://github.com/saghen/frizbee/commit/9b48b0947f1116fa676d6add16c787227a5991ea))

### Documentation

- add television and atuin to readme! ([3fb4793](https://github.com/saghen/frizbee/commit/3fb479322a411396cc1cb71ca046a70534e2e1b3))
- fix prefilter haystack mask examples ([2402db5](https://github.com/saghen/frizbee/commit/2402db569aa2ecb54080f83efa6637081fce694f))

### Performance

- single gather vec for multi-pattern matching ([02faa31](https://github.com/saghen/frizbee/commit/02faa3137a0d319f0c9f32895abebb8e9346a210))

## New Contributors ❤︎

- @ryanmab made their first contribution in [#88](https://github.com/saghen/frizbee/pull/88)

## [0.11.0](https://github.com/saghen/frizbee/compare/v0.10.0..v0.11.0) - 2026-07-13

### Breaking Changes

- switch to `u32` indices ([2532849](https://github.com/saghen/frizbee/commit/2532849b0a358b3ebcac2aee6445e7eb1e5823b3))
- `Pattern` and multi-pattern matching ([bbcfd0e](https://github.com/saghen/frizbee/commit/bbcfd0ed6d6ee275008c359058ced682736505f8))
- drop top level `match_*` functions ([28688b5](https://github.com/saghen/frizbee/commit/28688b59d03aecda3bba19aada3f83d0caf9ba50))
- rework `SortStrategy` with reversed variants ([6831669](https://github.com/saghen/frizbee/commit/6831669ab4081e01df3375aa0eee73ede19be713))

### Features

- prefix, suffix, substring, exact matching (#86) ([b55fad6](https://github.com/saghen/frizbee/commit/b55fad6ea86e05792d180b2056386fad10c71f79))
- `SortStrategy` enum, fix parallel ordering ([a7d456e](https://github.com/saghen/frizbee/commit/a7d456e2a1980c6a8a4c49efdfbc79b9483cc51c))
- limit threads based on haystack count ([af0634a](https://github.com/saghen/frizbee/commit/af0634a40285b4209c43093bb87dfd8e40e1d8fd))

### Bug Fixes

- arm64 load_partial on len 8 ([022a5ae](https://github.com/saghen/frizbee/commit/022a5aea98701bba2c106200b80fb600fe40fdbc))
- prefix bonus applied due to prefilter substr ([3b9ba7b](https://github.com/saghen/frizbee/commit/3b9ba7b70548c7daea1214840d40d85a89499216))
- ensure parallel matcher sorts by index ([41fc7d5](https://github.com/saghen/frizbee/commit/41fc7d5ea902468626b1a191fdf463156ae5d19c))
- remaining uses of top level match_* ([96ea2fa](https://github.com/saghen/frizbee/commit/96ea2fa37886a5045c4cc1c213263ce88708cd6c))
- match_greedy prefix bonus ([da380f5](https://github.com/saghen/frizbee/commit/da380f5243ee32117e01ac306b1d512b31549673))
- escaped ! on query pattern ([d4eced5](https://github.com/saghen/frizbee/commit/d4eced5f9efa6ba1e427da000fb7baa30f87d98c))

### Documentation

- k_merge module ([7c9b262](https://github.com/saghen/frizbee/commit/7c9b26267716610bde1803518074bda684c671a4))

## [0.10.0](https://github.com/saghen/frizbee/compare/v0.9.0..v0.10.0) - 2026-07-03

### Breaking Changes

- simplify matcher dispatch, drop some APIs ([364fc0e](https://github.com/saghen/frizbee/commit/364fc0ea4fa5dd685940ee3e35e68bc340f170cb))
- drop some APIs, add miri + proptests ([fc16a54](https://github.com/saghen/frizbee/commit/fc16a54a1e724383293e071c65a8d1dc06d48bac))
- `*_matches` on `radix_sort`/`k_merge` ([1e4267b](https://github.com/saghen/frizbee/commit/1e4267b5af0a9361c9eb60003dc347079d7539eb))
- many rare panics and typos ([9c3b8dd](https://github.com/saghen/frizbee/commit/9c3b8ddc3b0bca81aea9622814e8be0bf6f882c1))
- rename UnicodeMatching::Respect to Always ([9024633](https://github.com/saghen/frizbee/commit/9024633830999bb0ec79c9df105b3a60eab193e2))

### Features

- u8 smith waterman scoring ([2e5a912](https://github.com/saghen/frizbee/commit/2e5a9123571c972f165785ce598012d684c51b06))
- avx512 smith waterman ([eaff2fe](https://github.com/saghen/frizbee/commit/eaff2fee7aaf9284946e36afaeab30448898f446))
- avx512 u16 smith waterman ([ec2a8f6](https://github.com/saghen/frizbee/commit/ec2a8f698dfac988c292d9c796c3042ac3655e16))
- switch prefilter skipped_chunks to skipped_chars ([0b18811](https://github.com/saghen/frizbee/commit/0b188116e3b89093111cc14f07ff455667bdbbe7))
- avx512 prefilter ([651b4e1](https://github.com/saghen/frizbee/commit/651b4e16a1fda02070fc1d173867c119334422ab))
- rewrite prefilter with generic backend ([457c0ce](https://github.com/saghen/frizbee/commit/457c0ce46074e417869d728889a56b1bec6d1b4e))
- smart and case sensitive matching ([32c5ff3](https://github.com/saghen/frizbee/commit/32c5ff3bec10a04e460c22c4cdca44482b9a4074))
- custom k-way merge, drop itertools ([e8237bb](https://github.com/saghen/frizbee/commit/e8237bb1a9b8caed45ff4d17a0491f92244113ac))
- add back set_* matcher APIs ([0540458](https://github.com/saghen/frizbee/commit/05404589214206c8a7d3a5300559e13018c56f14))
- replace raw_cpuid with is_x86_feature_detected ([edcddf6](https://github.com/saghen/frizbee/commit/edcddf660b3ec0b8d169d6f2235f4015e5c99648))
- add fzf bench via std::thread::sleep ([9004beb](https://github.com/saghen/frizbee/commit/9004bebd02c9c1f549683d0a0a2a5c73da4f10e6))
- unicode prefilter, drop fuzzing ([36befb8](https://github.com/saghen/frizbee/commit/36befb8eeb5d4eb76d33f8e1e6f76be5d70d3f11))
- add unicode arabic/korean benches ([4961de0](https://github.com/saghen/frizbee/commit/4961de0bfb862758dffcb67ff324e32deca76486))
- rework unicode approach ([3eb7cbf](https://github.com/saghen/frizbee/commit/3eb7cbf0fd3dc4f71b1b33b2b7cd717098154c13))
- avoid updating matcher if nothing changed ([e4fc06c](https://github.com/saghen/frizbee/commit/e4fc06c5befe57c9c1ae179eea1d2ff03cee7cb9))
- unicode prefilter typos ([0123732](https://github.com/saghen/frizbee/commit/01237328f5e1d8dd6445e07544e3043a8d7a4883))
- support unicode `CaseMatching::Smart` ([06c9494](https://github.com/saghen/frizbee/commit/06c9494e24a21f135c8a8379d9441cfe56545533))
- case insensitive unicode matching ([88459c2](https://github.com/saghen/frizbee/commit/88459c27863e8fdf5d0126ebc63c0acfa3035b3c))
- unicode smith waterman ([0bf1fa8](https://github.com/saghen/frizbee/commit/0bf1fa8f97977a3a3fa4773cce4d6fb3b0428117))
- `UnicodeMatching` enum to control unicode behavior ([ca2f547](https://github.com/saghen/frizbee/commit/ca2f54742658cf3fc57915f957db0d3e377f0c2f))
- swap out bolero for proptest ([5544781](https://github.com/saghen/frizbee/commit/5544781916f26ba6093aa239abb816344b4a20ed))
- `match_iter` API, `fuzzy_match` iter ext ([af7cebf](https://github.com/saghen/frizbee/commit/af7cebf5dd1ea7a770adbb36a08ab9805a9f124f))
- add `match_list_parallel` to `Matcher`, restructure ([4fd9590](https://github.com/saghen/frizbee/commit/4fd9590c5a1dc27bf9d22f4fe4159a5335c82fa1))
- `Matcher::match_one` and extensive docs ([c778119](https://github.com/saghen/frizbee/commit/c77811977e6e4ae0b1b3514995289dda9116254a))
- simplify nix flake ([e4f9dba](https://github.com/saghen/frizbee/commit/e4f9dba4c67352df79016f46ef02b991689d70ec))

### Bug Fixes

- match_end_col sort tests ([ae9d0b6](https://github.com/saghen/frizbee/commit/ae9d0b6d41631ca3b314bd4fd9a9db1fb3132e17))
- gate prefilter overread behind safe_read (#80) ([7892da7](https://github.com/saghen/frizbee/commit/7892da7a82eecd95ad6e2643c5bdd628a7611058))
- prefilter typo chunk boundaries ([9312dce](https://github.com/saghen/frizbee/commit/9312dce69d98685c427a45f78eba739451353c08))
- match_end_col sort bench ([599c950](https://github.com/saghen/frizbee/commit/599c950daee9560b7cc2306fff0bfefa11f95440))
- keep `match_list` and `match_list_indices` filtering in sync ([34c8bac](https://github.com/saghen/frizbee/commit/34c8baca08ed89bc7958c4d191a07835034289e1))
- assert nucleo returns no intermediate snapshots ([98139d9](https://github.com/saghen/frizbee/commit/98139d9ca1b0cf8a89552c677c7fa8b3c7efd4fb))
- multi-byte unicode char incorrect indices ([c853a1b](https://github.com/saghen/frizbee/commit/c853a1b8ac3a14ab9853a1b68b84f21705ac35ae))
- incorrect `score_fits_in_u8` calculation ([59a17d8](https://github.com/saghen/frizbee/commit/59a17d8118db7f8f74da8874996941e63da539f3))
- `match_one` not inlined ([bd87378](https://github.com/saghen/frizbee/commit/bd87378fb45548bbcb60e318b5692c898a941211))
- arm neon movemask ([35208a8](https://github.com/saghen/frizbee/commit/35208a8134e1dd62c3675031b303d919125dd261))
- unicode incorrect indices ([c9ee4b3](https://github.com/saghen/frizbee/commit/c9ee4b31f747c145d796a1cd4a751c2d1a34fe0d))
- many typos, miri bug, naming, docs ([14d2f19](https://github.com/saghen/frizbee/commit/14d2f1954d8481c5569ff25527ffc0f7595041f1))
- match delimiters between greedy and sw ([9cb9979](https://github.com/saghen/frizbee/commit/9cb99795824cfeaa6c9ec4b78ad0ba1c4503b4b6))
- incorrect alignment indices on unicode ([e5c7bc5](https://github.com/saghen/frizbee/commit/e5c7bc522e25984a464de6ad59bf6eea50e19239))

### Refactor

- move sw backends under sw::simd::backend ([441b78c](https://github.com/saghen/frizbee/commit/441b78c25ee4a55bbc5362d833f52c682df7d211))
- move prefilter backends under prefilter::backend ([523d9eb](https://github.com/saghen/frizbee/commit/523d9eb7610ef21b4951d3ba0bd9f75f57e3947e))
- move prefilter algorithm into algo ([121851f](https://github.com/saghen/frizbee/commit/121851f516592ff860e84ec35a176c3bc16d8e6a))
- specialize matcher by backend ([e582817](https://github.com/saghen/frizbee/commit/e58281770420c56de1dcf9e4804ecdd24c93506f))
- simplify prefilter wrappers ([9ce2b5e](https://github.com/saghen/frizbee/commit/9ce2b5e401c677836acd397336a10bcba270ba58))
- simplify smith waterman wrappers ([8d10c81](https://github.com/saghen/frizbee/commit/8d10c81fb4b6b0fdc8fa22c0e9c6f968877a04b9))
- move `smith_waterman::simd` to `smith_waterman` ([393aed0](https://github.com/saghen/frizbee/commit/393aed007c6383326dd7c0c7e087b0e06c390ed2))
- create `Matcher` once in benchmarks ([9af3b6c](https://github.com/saghen/frizbee/commit/9af3b6ca8865c05996cd0f11ba359bf98d164ea1))
- drop prefilter `Backend::broadcast` ([3c386b2](https://github.com/saghen/frizbee/commit/3c386b2242e149d169b1e6ca87f1f0edcdf070fa))
- unicode matching helper, backwards scan ([2ed88a4](https://github.com/saghen/frizbee/commit/2ed88a4b450a8d66a7bc2c546f70b1bbbd823f36))
- split up prefilter/algo file ([eb60749](https://github.com/saghen/frizbee/commit/eb60749ec163b26504713d761b255c8627e336a7))
- move gap impls to `smith_waterman::algo` ([6297c90](https://github.com/saghen/frizbee/commit/6297c901a67ddfc5d9b1b966634ee511b5814084))
- fix missing `self.needs_unicode` case ([c03999a](https://github.com/saghen/frizbee/commit/c03999a513ad044c6658a522592c9abb20b7d538))

### Documentation

- misc rewriting and cleanup ([119b78c](https://github.com/saghen/frizbee/commit/119b78c7ed552da42c831c55849a3effb5c28ee8))
- update benches and prefilter docs ([dd7bbf6](https://github.com/saghen/frizbee/commit/dd7bbf6d43d22867350eb9a710893a8d00c9357d))
- update benchmarks ([c56dac9](https://github.com/saghen/frizbee/commit/c56dac995df416a69155abeb7883c6f6a9f2a470))
- unicode ([5ee37ee](https://github.com/saghen/frizbee/commit/5ee37ee6d49c0d62d2a33b8ea574ad693e363470))
- mark unicode gap docs as text ([3aeb1dc](https://github.com/saghen/frizbee/commit/3aeb1dc18a645ec329284d3fc54fa3774e983c50))
- update benchmarks ([50c718f](https://github.com/saghen/frizbee/commit/50c718f5f326d47f947bc5f099075ac2a3a1b988))
- add note about NEON prefilter performance ([7f8defb](https://github.com/saghen/frizbee/commit/7f8defbec65645aca511347a2333bfbf9e1e423f))
- match lib.rs docs to README ([7c8d7e0](https://github.com/saghen/frizbee/commit/7c8d7e07220e4e9d6cdc1512053fc2ff7471d9fe))
- misc typos and renames ([6edeb89](https://github.com/saghen/frizbee/commit/6edeb897d1d1984fcceed8ec55b0863b16a63709))
- update benchmarks ([899c473](https://github.com/saghen/frizbee/commit/899c473cdf620e72ccd35184a1db7ca0051f2877))
- add keywords/categories ([e87e0bd](https://github.com/saghen/frizbee/commit/e87e0bd7f9aee29d856d1a8ea56946e349e94ec0))
- proper casing for nucleo/fzf ([d5f9282](https://github.com/saghen/frizbee/commit/d5f928265d600626806ddc1e292b5009ddf48113))
- drop number for fuzzy_match slowdown ([c56f614](https://github.com/saghen/frizbee/commit/c56f6142898af6f9af1c20f8be725db8c269ec74))

### Performance

- avx512 shift right lanes ([0287fd5](https://github.com/saghen/frizbee/commit/0287fd58818d260c4ea5b157122858105d1a1db7))
- trim prefilter match windows ([b1cbddf](https://github.com/saghen/frizbee/commit/b1cbddf658056e570c10531b11fe5ee9419d20a1))
- specialize avx2 prefilter no typo ([6923519](https://github.com/saghen/frizbee/commit/692351960b7c202c96b2072385e9b6d92a774afb))
- specialize avx512 prefilter typos ([d5b4c57](https://github.com/saghen/frizbee/commit/d5b4c5749aa45f59f02e3b7468b5faaded931330))
- add avx512 prefilter masked loads ([4f2d57a](https://github.com/saghen/frizbee/commit/4f2d57aee7d7d5af152934f61c97a59e59eb9667))
- move target features to matcher dispatch ([5c9145b](https://github.com/saghen/frizbee/commit/5c9145b1d82ec1ef88bb8f3b7298769b0b4f0558))
- increase parallel chunk size ([88f52fb](https://github.com/saghen/frizbee/commit/88f52fb231f2ab99ade7935f011d3f7f6a8a01aa))
- avoid sorting on empty needle ([114d188](https://github.com/saghen/frizbee/commit/114d188948cf944bc6ec914abe8c441a2bc25646))
- per-specialization hot loops ([82bae12](https://github.com/saghen/frizbee/commit/82bae1294422bf7d2a641a1f7f1320fdbf92b759))

### Testing

- add prefilter backend parity cases ([32436c5](https://github.com/saghen/frizbee/commit/32436c5767e3e6307050814e4415bdae5de72374))
- swap proptest for bolero, integration tests, expand suite ([d2941c3](https://github.com/saghen/frizbee/commit/d2941c3ece81e99b153518141256b706ace9a755))
- fuzzing ([55ea87c](https://github.com/saghen/frizbee/commit/55ea87c94bebb3b61affa9b4807b55428a9b2ecf))
- extra unicode cases ([75ccfe1](https://github.com/saghen/frizbee/commit/75ccfe18b64f2a6dbe9a9ae828d333a2a8345edf))
- share generator, simplify sw backend tests ([c248438](https://github.com/saghen/frizbee/commit/c248438626db59ec0b2fe214e6cd3a1cda4afb14))

## New Contributors ❤︎

- @leonardcser made their first contribution in [#80](https://github.com/saghen/frizbee/pull/80)

## [0.9.0](https://github.com/saghen/frizbee/compare/v0.8.3..v0.9.0) - 2026-04-04

### Features

- add scalar implementation fallback (#76) ([3aff5b5](https://github.com/saghen/frizbee/commit/3aff5b533a9a02afca8647823778da3cbdd25ee8))
- track match end column position via `match_end_col` feature (#75) ([53be326](https://github.com/saghen/frizbee/commit/53be3268b295b2189c7e1807e9a93b50b85f447b))

### Documentation

- update fzf comparison on docs.rs ([8b59da3](https://github.com/saghen/frizbee/commit/8b59da305dc96170a9de4a9646e5c17efdf88efd))

### Performance

- use radix sort for sorting `Match`es (#74) ([8b4d19c](https://github.com/saghen/frizbee/commit/8b4d19c9b4395019390692803727b138504ca5d7))

## New Contributors ❤︎

- @Xeonacid made their first contribution in [#76](https://github.com/saghen/frizbee/pull/76)

## [0.8.3](https://github.com/saghen/frizbee/compare/v0.8.2..v0.8.3) - 2026-03-10

### Features

- Optimize latest rewrite (#72) ([cd448bd](https://github.com/saghen/frizbee/commit/cd448bdb9836f98eb278ac20b852801d08d4513a))

### Bug Fixes

- zero out missing null matrix rows ([a672ae8](https://github.com/saghen/frizbee/commit/a672ae8ca1fa164dc35458ba74d177d6b6718ecf))
- dont skip prefilter for max typos >= 3 in sse ([f2ab3fb](https://github.com/saghen/frizbee/commit/f2ab3fb1f685a7764405f4ef6324fadfbd9090d8))
- 256-bit SSE smax_u16 combining 8 bit lanes ([b64cdd2](https://github.com/saghen/frizbee/commit/b64cdd2d08ccca7c766c26044dc1e8ea966e4500))

### Documentation

- drop fzf comparison ([4862d4a](https://github.com/saghen/frizbee/commit/4862d4a545a213edcaad23675aec34a416f1377c))
- fix fzf bench, thanks @junegunn ([edf7a58](https://github.com/saghen/frizbee/commit/edf7a58c86220ba897fbdd1a961768ccdd97dc38))
- fix fzf bench spacing and formatting ([efc3f88](https://github.com/saghen/frizbee/commit/efc3f88ae4ad5535a799803e99cc6de5c41d05b6))
- fix readme example ([6c69acd](https://github.com/saghen/frizbee/commit/6c69acd17e787fe0949aece3ceea11ffc10f7528))
- update fzf benchmarks ([97ac04d](https://github.com/saghen/frizbee/commit/97ac04d4a2d486f05982f99b5c3fcec707732579))

### Performance

- drop a few instructions for gap extension ([5bec7b1](https://github.com/saghen/frizbee/commit/5bec7b1b6abd7710821c5d5240cfd356ad60f8e3))

## [0.8.2](https://github.com/saghen/frizbee/compare/v0.8.1..v0.8.2) - 2026-02-20

### Bug Fixes

- guard against overflow on `score_haystack` ([b972e79](https://github.com/saghen/frizbee/commit/b972e79ca3eb4d764e598ab3f0c22ccb20102a62))
- dont skip chunks when typos are allowed ([aaaa652](https://github.com/saghen/frizbee/commit/aaaa6521ec20b9619d1f00eb326f37d78d61a981))
- zero out null matrix rows ([7162754](https://github.com/saghen/frizbee/commit/71627549f559bf1b9ac273af91ffce6a7c5228b4))

### Documentation

- misc clarifications ([fb11ae3](https://github.com/saghen/frizbee/commit/fb11ae3f9b36fd7bb6daad62a167e7cee1757049))
- rename one shot and parallel benchmarks ([e4ac174](https://github.com/saghen/frizbee/commit/e4ac17481d2a00cbc9254dd04798d807cba459e1))
- update benchmarks ([cc93ef2](https://github.com/saghen/frizbee/commit/cc93ef200f48ebf61228ec5f33dc4f2c4053d546))
- add fzf benchmark ([1d8c210](https://github.com/saghen/frizbee/commit/1d8c21071dd75cbaea39d1eab7d3d310acf28d9b))

## [0.8.1](https://github.com/saghen/frizbee/compare/v0.8.0..v0.8.1) - 2026-02-19

### Refactor

- drop reference implementation ([7a4b861](https://github.com/saghen/frizbee/commit/7a4b861920bda21c78cd62bc7833526074ae8c1e))

### Documentation

- alllll the docs ([78b62c4](https://github.com/saghen/frizbee/commit/78b62c4105c56a1843a477148a59f868e5ae65a8))

## [0.8.0](https://github.com/saghen/frizbee/compare/v0.7.0..v0.8.0) - 2026-02-19

### Features

- intra sequence smith waterman ([bbcf283](https://github.com/saghen/frizbee/commit/bbcf2834e7c01bc80cfcdfd6565d7e8c9051d1ec))
- matcher, many bonuses, prefilter perf ([cfca682](https://github.com/saghen/frizbee/commit/cfca68273a8ecfcb12c720cd3f84657256a9aa5a))
- drop v1 and nightly, adopt v2 ([ae0fcdd](https://github.com/saghen/frizbee/commit/ae0fcddd1ef7fc65e7eb1ede2bf0c259ecb196dc))
- move matchers into structs, add parallel matching ([eb0aa7f](https://github.com/saghen/frizbee/commit/eb0aa7f19ce907233a5f4150fc1d0531a11f7af0))
- runtime feature detection for prefilter, reorganize matcher ([4c9ac62](https://github.com/saghen/frizbee/commit/4c9ac627e3802973470ad9eeb404bb170be3cefb))
- flatten score matrix, wip incremental matcher ([bf4ccae](https://github.com/saghen/frizbee/commit/bf4ccaee086d647aea6ce6fff186f73207ceb8b1))
- generic SIMD vectors ([dc9cb6d](https://github.com/saghen/frizbee/commit/dc9cb6d3ec576c3a254bb5adb49045a250199eb4))
- a very slow incremental matcher ([6c20e92](https://github.com/saghen/frizbee/commit/6c20e92c528fdce308ab48559cfc1f2923783d5a))
- double pumped SSE256, runtime feature detection, drop aligned ([f99eb90](https://github.com/saghen/frizbee/commit/f99eb90fb6b7e74b9c5389beba01a8bd2a44e47c))
- vector tests, drop par sort, assert needle len ([9d34e4b](https://github.com/saghen/frizbee/commit/9d34e4b303ce592dc6fe80ec6e0219636a3cc5ba))
- aarch64 support, matrix fixes, prefilter runtime feature detection ([6877319](https://github.com/saghen/frizbee/commit/687731962a7a68f72213b2209bd1f633dc3ed05a))
- test idx_u16 implementation ([777ed94](https://github.com/saghen/frizbee/commit/777ed9434c9cb739514bb4803ef91688b5f91781))
- guard against typos leading to all matches ([f1d6171](https://github.com/saghen/frizbee/commit/f1d6171c00f98675d94d2722a66c13ae6238ec89))
- drop nightly toolchain!! ([86c54f0](https://github.com/saghen/frizbee/commit/86c54f04ca925e91dbada478108b170c1b701859))
- indices and alignment api (#61) ([8a8a5be](https://github.com/saghen/frizbee/commit/8a8a5be1d272c9029360517bb23ce95795e12754))
- add `match_iter` and `match_iter_indices` to `Matcher` ([96581ca](https://github.com/saghen/frizbee/commit/96581ca5774b7b093f096b1d3ff95b437511c5ed))
- add 2 typo and 3 typo benchmarks ([5432842](https://github.com/saghen/frizbee/commit/5432842b67c879771dc3505403e336c52e786605))
- skip prefiltering for typos >= 3 ([5e996de](https://github.com/saghen/frizbee/commit/5e996de89a9008ecef3c7aa61d85849693e0afeb))

### Bug Fixes

- attempt to subtract with overflow (#55) ([4abc19d](https://github.com/saghen/frizbee/commit/4abc19d0dc7a7bf927fde1e727dc94ca0e1f2b9e))
- prefix score bonus ([22403bf](https://github.com/saghen/frizbee/commit/22403bf3bc8c1128493b0334c7fb6119038f0cf4))
- gaps not propagating past 16 bytes ([b4ffb66](https://github.com/saghen/frizbee/commit/b4ffb6684e47a5cdd708dcf6cc40bd0106159ec2))
- dont sort parallel when sort is false ([563e8ee](https://github.com/saghen/frizbee/commit/563e8eed5ca6ae329306f33d62b55974d0ad4a26))
- store and use adjacent match mask, minor gap perf ([114218d](https://github.com/saghen/frizbee/commit/114218d32a41dde218e6e2bbe3a37566d7d81362))
- max typos off by one ([85e6bba](https://github.com/saghen/frizbee/commit/85e6bbabe2e7edbda0bafc4c48dcfb85a5eef8cf))
- drop guard against typos leading to all matches ([4a6eb14](https://github.com/saghen/frizbee/commit/4a6eb14f7a2cbb080cda926663b22d11064624e7))
- matrix struct, use match masks in alignment ([2bf5878](https://github.com/saghen/frizbee/commit/2bf587830799f16037b25c2cb4beb83f15342777))

### Refactor

- misc ([cf2b0a4](https://github.com/saghen/frizbee/commit/cf2b0a45849130b7ea44f62a5749c8c755cf6a97))
- rename internal match list ([4286659](https://github.com/saghen/frizbee/commit/42866598a61aea6c5f30d21bbf73f58019a19936))

### Documentation

- update prefilter safety ([7385671](https://github.com/saghen/frizbee/commit/738567138d1204c5fea8b810c35815c3ad7e8c43))
- update readme and benchmarks ([0e3033e](https://github.com/saghen/frizbee/commit/0e3033eb294cc52675ade9b1982feb14d3cf2066))
- fix skim link ([04256c0](https://github.com/saghen/frizbee/commit/04256c071648a49a63e6947bbfa641a1ddeeafc4))
- `match_iter` example ([da481cb](https://github.com/saghen/frizbee/commit/da481cb613829dc0e957e6dee03ff0d58212079b))
- minor edits to smith waterman section ([59e3e6e](https://github.com/saghen/frizbee/commit/59e3e6ebf1fb2eb2e6e5dce6f3f2e611099b3059))
- clarify overlapping load ([b08f281](https://github.com/saghen/frizbee/commit/b08f281a40ed462988142301a45b811dbe25fabc))

### Performance

- preprocess bonuses ([5cf59a2](https://github.com/saghen/frizbee/commit/5cf59a22be59214b1d373ea4c744effe3890c5c5))

## New Contributors ❤︎

- @raui100 made their first contribution in [#55](https://github.com/saghen/frizbee/pull/55)

## [0.7.0](https://github.com/saghen/frizbee/compare/v0.6.0..v0.7.0) - 2026-02-03

### Bug Fixes

- build on latest nightly (#54) ([45dd259](https://github.com/saghen/frizbee/commit/45dd259505357c6efbe8a4d8e213b918d867f884))

## [0.6.0] - 2025-11-14

### Breaking Changes

- drop `min_score` option ([8d7c445](https://github.com/saghen/frizbee/commit/8d7c445ce243364c6e6339cf3db646467f4323fc))
- `Options -> Config`, `Scoring` options, `index_in_haystack -> index` ([539cbef](https://github.com/saghen/frizbee/commit/539cbef635c5d36114bca8876aa67f9e0ccd78b7))
- remove unused modules from api, docs ([e5ad501](https://github.com/saghen/frizbee/commit/e5ad501c5767ee913b1fe1745f8749de6872da34))
- use ref borrowing for config (#47) ([b7f2b5f](https://github.com/saghen/frizbee/commit/b7f2b5fba871db387e6931ea6feaeb9eb4ad0f3b))

### Features

- initial ([2adc343](https://github.com/saghen/frizbee/commit/2adc3439311e43864506878e3250e1e440e20ff6))
- cleanup and rename ([8bc2eee](https://github.com/saghen/frizbee/commit/8bc2eee0c65da5b011b71b270d19b3ed22262a97))
- add index_in_haystack to match ([50a22da](https://github.com/saghen/frizbee/commit/50a22da50dac82b9a9e97c56c95fdfa5a9737a55))
- reduce prefix bonus ([2cc6511](https://github.com/saghen/frizbee/commit/2cc65117aa2c9cbc24cb77bd649caabffd6c8de0))
- adjust scoring to encourage delimited/capitalized matches ([d4c715d](https://github.com/saghen/frizbee/commit/d4c715df5b08f70d91b33cb1c7c1eff06a2e5ee0))
- temporarily disable capitalization bonus ([3e7d321](https://github.com/saghen/frizbee/commit/3e7d32108229f220bb25e3416e36ffc8c2750166))
- exact match bonus ([1773ea4](https://github.com/saghen/frizbee/commit/1773ea4bbba2b703150166654296a8f4dd1e7adc))
- matching case bonus, delimiter fixes ([a5089ca](https://github.com/saghen/frizbee/commit/a5089ca087eca2e538bf6e58a995816a3f0d103d))
- tests ([c25a00a](https://github.com/saghen/frizbee/commit/c25a00ab0f1b4946d5ba2ab2eed3da5c9fd139eb))
- drop useless simd_max(zero) ([8c3037f](https://github.com/saghen/frizbee/commit/8c3037f0d69cd62130017006d8f78cf65bebb5d1))
- drop useless select and simplify code (#3) ([f053b13](https://github.com/saghen/frizbee/commit/f053b1344a6501e70c621dc4f56d14a274596ac5))
- enforce 128 bit SIMD, avoid 256 bit ([0d42d37](https://github.com/saghen/frizbee/commit/0d42d37720a85d6b2fd5b51038a3b94978e88b46))
- matched character indices ([d632d11](https://github.com/saghen/frizbee/commit/d632d11132bb0acd99a5d91310149c4422ea225c))
- add delimiter bonus on colon ([78aa772](https://github.com/saghen/frizbee/commit/78aa7729f29ac358dbd5553965138ad542bae725))
- add debug, clone, copy to options ([7c2aa46](https://github.com/saghen/frizbee/commit/7c2aa4661a43c6f8565c2200a07428bdf675ce1b))
- implement smith_waterman using a generic function (#6) ([8f174de](https://github.com/saghen/frizbee/commit/8f174de75faee8fb901fe9c27c0386e6450c7d61))
- wip score matrix ([df9107e](https://github.com/saghen/frizbee/commit/df9107e475ccd9a63b9f2cfea3d4d505ab331f07))
- non-simd reference implementation ([a960f12](https://github.com/saghen/frizbee/commit/a960f12704ddaa6a35c41b04967c27c66231d75a))
- add iai benchmark ([0ea8d8f](https://github.com/saghen/frizbee/commit/0ea8d8fb10dc0147ca5df54a5cc5b67d6b3d5951))
- alignment for typo count ([4016d65](https://github.com/saghen/frizbee/commit/4016d65dee8a09b514092631c99dea1c113ed8a7))
- add debug trait to SimdNum ([5ecc9ba](https://github.com/saghen/frizbee/commit/5ecc9ba6bb2a77760ac16f077e849139faf0a208))
- prefilter with typo proof of concept ([65ce6aa](https://github.com/saghen/frizbee/commit/65ce6aa32f3461c82277ba4716a4c88397349b71))
- rework benchmarks ([e798541](https://github.com/saghen/frizbee/commit/e7985412acb23068f7c6647e9a2bd268173a1032))
- update reference implementation ([0e1d186](https://github.com/saghen/frizbee/commit/0e1d186f28bb2402a280e32042bf76c5fc587079))
- rework iai benchmark ([0abc726](https://github.com/saghen/frizbee/commit/0abc726eaf7ecb776f3194f9bdc57b979b069f23))
- replace prefilter with max typos option ([da006cc](https://github.com/saghen/frizbee/commit/da006cc7b9616e86a8990b6a32940e2bb79825d6))
- filter by max typos, minor perf improvements ([fa3bd32](https://github.com/saghen/frizbee/commit/fa3bd32ba64cd1fa2ca4158c376612c2ea4c4d10))
- bitmask prefilter ([6d08a89](https://github.com/saghen/frizbee/commit/6d08a8961836bb2b059ddfb403d71f1aff433e5d))
- indicate exact matches (#14) ([75c9dc5](https://github.com/saghen/frizbee/commit/75c9dc534f459580dba8a45847bfae2e384f1a54))
- drop `memchr` dependency ([bf27dba](https://github.com/saghen/frizbee/commit/bf27dba56fa3c1147ea65cfc947d40318cda9f7d))
- update rand/rand_distr ([283e4c4](https://github.com/saghen/frizbee/commit/283e4c444fadd3d4459c31c01128123ffc0a6b25))
- increase score bonus for matching case ([6415483](https://github.com/saghen/frizbee/commit/6415483c58742ef66f1fd5e19ee4d122f0eb0534))
- increase max length to 1024 ([1a2a201](https://github.com/saghen/frizbee/commit/1a2a201ca59563e289d4a3abd3e77233db347d92))
- reintroduce prefiltering when haystack is too long ([1c86f18](https://github.com/saghen/frizbee/commit/1c86f18d2ca94fa2481f2ba13e1dc5d922f3800c))
- prepare to move smith_waterman inner loop into its own function, and for the incremental matching API ([be141ae](https://github.com/saghen/frizbee/commit/be141aebfd471e54b33e81faf741da72c4eb742e))
- enable capitalization bonus ([5ba2cbc](https://github.com/saghen/frizbee/commit/5ba2cbc8c907b2cc2ca479f0fd506c822d402412))
- incremental matching api ([2e1c635](https://github.com/saghen/frizbee/commit/2e1c6358b08f66ceed2e8c652a6477a3c9088e35))
- compute all_time_max_score after score_matrix ([8e694d6](https://github.com/saghen/frizbee/commit/8e694d6806d65773a5f148b6c364700a765e054e))
- add benchmark ([af6623c](https://github.com/saghen/frizbee/commit/af6623cc6d7c0485961c0cc5620cb68ddc4ff8d6))
- avoid sorting for incremental matcher ([63711eb](https://github.com/saghen/frizbee/commit/63711ebd0361b35430d25698a5d5bd77a53a4469))
- restructure codebase ([4cc3699](https://github.com/saghen/frizbee/commit/4cc369932d8968e23cfe7dd1d7f58fce811aeac2))
- use case matching ignore for nucleo benchmark ([14718d6](https://github.com/saghen/frizbee/commit/14718d6225bfd17f567807619cd598f3ed8d41a6))
- lazy and stack allocated bucket initialization ([8c7686d](https://github.com/saghen/frizbee/commit/8c7686d62540e2be2e4418ded13c430996bf23f4))
- drop lazy bucket initialization ([1b5ea29](https://github.com/saghen/frizbee/commit/1b5ea297b3eb20b3fbda1673823eb13ec65e8dfa))
- increase benchmark sample count from 1000 to 10000 ([86b515e](https://github.com/saghen/frizbee/commit/86b515e6c642422e14c5fa8da0ecba3d6e2d85c9))
- expose `IncrementalMatcher` and `match_list` ([6329b44](https://github.com/saghen/frizbee/commit/6329b4400375d2156db74f50bb13102f002216ea))
- add serde `Serialize` and `Deserialize` derives ([a472695](https://github.com/saghen/frizbee/commit/a472695c073cc9aba8a2e4fb7a228b3c3ab56b8f))
- support any string type ([9007ef1](https://github.com/saghen/frizbee/commit/9007ef1aaba8cf2bbff29995a95292169b5e663d))
- no delimiter bonus for delimiter chars ([d96d203](https://github.com/saghen/frizbee/commit/d96d203db439a3b54c4970823378bc51cb4f6fe9))
- split up simd code, add matched indices support ([2c96213](https://github.com/saghen/frizbee/commit/2c962137c3530bfd786e35912c82f84b299b177c))
- runtime instruction detection ([44482c3](https://github.com/saghen/frizbee/commit/44482c36f35ef6e60fff4837cb6a4d449efbabd8))
- decrease gap penalties ([4a1d113](https://github.com/saghen/frizbee/commit/4a1d11352d7b6aa7048916d08161004dbe4f6cc1))
- probably broken skim benchmark ([d981215](https://github.com/saghen/frizbee/commit/d9812156936dbaa2df41d7f4ef2e11b8670b16c8))
- ignore haystacks shorter than needle - max typos ([bd9be77](https://github.com/saghen/frizbee/commit/bd9be7739600cab22ece25c7c09be731d384afe2))
- avoid processing unreachable matrix elements (#29) ([c23d1ab](https://github.com/saghen/frizbee/commit/c23d1aba9711bc83ddf92748d93babe0f239053e))
- stop calculating typos after reaching max typos (#30) ([0742f8c](https://github.com/saghen/frizbee/commit/0742f8cdef216d0790b6dcbe8273266c7cb78418))
- add greedy fallback matcher ([a308373](https://github.com/saghen/frizbee/commit/a308373446b57b6a7f62b2a138665c856a78f459))
- multithreading (#32) ([6fd0960](https://github.com/saghen/frizbee/commit/6fd0960757b1290cd4251a4f9a8afedbb7946c7b))
- flake dev shell with aarch64 and x86_64 targets ([b982c62](https://github.com/saghen/frizbee/commit/b982c620cb40bb33a9ca9dd2e4d4bee00d1f19ce))
- fallback to greedy matching on >512, rather than >1024 ([004cb63](https://github.com/saghen/frizbee/commit/004cb63694302026d7d2865a1316c4a93515c220))
- increase gap penalty, reduce delimiter/capitalization bonus ([7a8cd28](https://github.com/saghen/frizbee/commit/7a8cd284b2397ba7c0cbe61d5e11e4884dd90857))
- offset prefix bonus ([2701d9f](https://github.com/saghen/frizbee/commit/2701d9f4a868f95c6a2b12fd8f2969dd41d3cc05))
- drop `SimdNum` in favor of fixed `u16` score ([a51b58e](https://github.com/saghen/frizbee/commit/a51b58ee0a7d00f2b7528ac749878e999ec02ea0))
- SIMD string interleave ([fd31411](https://github.com/saghen/frizbee/commit/fd314111aa10f27f7aed76f6cbcb52cb8d864387))
- use `x86-64-v3` for benchmarking ([2744cfb](https://github.com/saghen/frizbee/commit/2744cfb3e554c8b9090e77eb35745f59b27dfa81))
- use multiversion, generic interleave ([1246f97](https://github.com/saghen/frizbee/commit/1246f973439bc228884dcc6541ee2ad940734e7a))
- multiversion bitmask, misc tuning ([50d1815](https://github.com/saghen/frizbee/commit/50d1815f92fb4a3c2f8dbdb8ca65bf913e32ef07))
- prefilter multiversion and cleanup ([06914dd](https://github.com/saghen/frizbee/commit/06914dd131ef00e6f32d25ed584f1c6486160e21))
- SIMD intrinsics interleave ([92f67d8](https://github.com/saghen/frizbee/commit/92f67d8478532a9e11915426abbca5f523ae230b))
- reference implementation offset prefix bonus ([afe55ea](https://github.com/saghen/frizbee/commit/afe55ea49278bc9112180edd23660e31cf23b38c))
- modularize and update benchmarks ([0e52f55](https://github.com/saghen/frizbee/commit/0e52f55de7ec60d733e5ebbad2ed1fd70d7eec80))
- upgrade to rust 2024 ([1c003ed](https://github.com/saghen/frizbee/commit/1c003ed229c8eb0e61b80d6340ef53e810bd880b))
- simd and intrinsics prefilter (#43) ([8637faf](https://github.com/saghen/frizbee/commit/8637faf38987196687e5a2e3c4f02792d295ec19))

### Bug Fixes

- *(bench)* return results from routine (#13) ([0bbeb41](https://github.com/saghen/frizbee/commit/0bbeb41c7d811eef94a346f4cad2ff410449719c))
- dont give capitalization bonus on first character ([284aa89](https://github.com/saghen/frizbee/commit/284aa89028ba552505892fec4d3aa593ddabc25a))
- add missing buckets ([3a8f894](https://github.com/saghen/frizbee/commit/3a8f8947551e1d86c51ea6ce24bc8b7adf7b6a31))
- add more smith_waterman_macro bucket (#1) ([d910bec](https://github.com/saghen/frizbee/commit/d910bec53b867ce06702520c7e05f9862bf78dd9))
- prefilter ([1f7d3f9](https://github.com/saghen/frizbee/commit/1f7d3f93f87e26c54762d6b563b47d18d3b3773d))
- prefilter not ignoring case ([5e6e4c6](https://github.com/saghen/frizbee/commit/5e6e4c65fbeb611a2ea0c694079202aba15915c1))
- bench crate name ([f2ae831](https://github.com/saghen/frizbee/commit/f2ae8310efe196772a9cc3d804eb5b4f54f88e3f))
- capitals on needle never matching ([36e564b](https://github.com/saghen/frizbee/commit/36e564b5ec00afcd16455d4bb9de27e1c205572f))
- add missing bitand to macro ([bba669b](https://github.com/saghen/frizbee/commit/bba669b4c8bc5bd92b527823dd291e3b7fc2413c))
- get current column scores as mutable ref ([570d68d](https://github.com/saghen/frizbee/commit/570d68dd38071f3af3e58e905acc89d7e13f6102))
- remove unnecesary conversion to u16 ([77f0ba5](https://github.com/saghen/frizbee/commit/77f0ba5a2236c640a6f81914f3b471ca533b39c1))
- properly ignore capitalization on prefix ([81be460](https://github.com/saghen/frizbee/commit/81be460e8cbd8cbb7d98fa77a61da35c247275f6))
- matched indices with typos ([a9d07fe](https://github.com/saghen/frizbee/commit/a9d07fe50e2f495e9957c3a593c57ba16e562f63))
- ensure minimum length of 1 for generate_haystack ([4107c75](https://github.com/saghen/frizbee/commit/4107c75f03dafbc9f3f4a21ce0bf41900d220dc4))
- ignore exact match test for incremental matcher for now ([330243e](https://github.com/saghen/frizbee/commit/330243ec5626894620559562ff84463e3bc54940))
- support different string types per arg ([74ff035](https://github.com/saghen/frizbee/commit/74ff035597a4a36776a10bc5d184898d20ed17ef))
- add dot delimiter to reference implementation ([31f9ae0](https://github.com/saghen/frizbee/commit/31f9ae02f355b2430614c86e97b1e22813da80a9))
- avoid capitalization bonus on capital after lower ([a84ebfe](https://github.com/saghen/frizbee/commit/a84ebfe06ada971a0b4a1a360e449de47bc27309))
- out of bounds and div by zero panics ([57bb136](https://github.com/saghen/frizbee/commit/57bb136f994ed3fdee9a2e4a564581743a22b841))
- conditional compile avx2 and avx512 paths ([6d0fa67](https://github.com/saghen/frizbee/commit/6d0fa67de8025fbc77078224be7131ccab8488cc))
- overflow on long needles with small haystack ([8de16e4](https://github.com/saghen/frizbee/commit/8de16e49910aafde4908db22e1c9d46d32f7da03))
- prevent thread count go to 0 (#41) ([cbf566d](https://github.com/saghen/frizbee/commit/cbf566d12595e6220ce24d0d688f43dd649a0bdc))
- ensure score matrix dimensions >=1x1 ([8c1153a](https://github.com/saghen/frizbee/commit/8c1153a159ae9aeda1ecf3c3dd9ab814c3ae60a3))

### Refactor

- *(bucket)* make generic on SimdNum ([a301a74](https://github.com/saghen/frizbee/commit/a301a74f90f0e1bf1a01ae48392944f57b48ef6e))
- *(smith_waterman)* traits instead of bounds ([b198250](https://github.com/saghen/frizbee/commit/b19825044aa9f5083d2ad7524065dca2be59981f))
- *(smith_waterman)* always return u16 slice ([1408b37](https://github.com/saghen/frizbee/commit/1408b3763b18f15e03f0c4aefcc6909184c2d15d))
- remove dead procedural macro code ([48f6c59](https://github.com/saghen/frizbee/commit/48f6c59ebee10ec783480a19f7ade2da722953ad))
- remove unused first char multiplier ([7e6e120](https://github.com/saghen/frizbee/commit/7e6e120b2a5108824552d1c87404e18900f6adf1))
- rename `curr_col_score_simds` to `curr_col_scores` ([16be4d0](https://github.com/saghen/frizbee/commit/16be4d025c11d2cdd9814aed26f7a56d00b6fef5))
- make smith_waterman generic on number of simd lanes ([c2fbb36](https://github.com/saghen/frizbee/commit/c2fbb360929ade066536359489c922bea8256a6d))
- impl simd traits with a macro ([2dfb23a](https://github.com/saghen/frizbee/commit/2dfb23ac4dbf041d5636373a5070d78e2f2dcfba))
- make constants u16 again (#12) ([096163b](https://github.com/saghen/frizbee/commit/096163b97af6e119b2b92db936af9165cdf9886e))
- clean up needle_len ([c23db08](https://github.com/saghen/frizbee/commit/c23db0827bcd5d1af6623a8036d3385cc256d799))
- minor cleanups / clippy ([2d33e60](https://github.com/saghen/frizbee/commit/2d33e6032c88e2495cc1da4a8a44072175eafbb8))
- move smith waterman inner loop into its own function ([59e6754](https://github.com/saghen/frizbee/commit/59e6754157bdf98e49c53e4460b132c0aef4e241))
- simplify prepare_haystack ([5709568](https://github.com/saghen/frizbee/commit/57095680387f6837e2a3d1f277457bb5f9f3d1c3))
- prefilter method selection ([6faf2f5](https://github.com/saghen/frizbee/commit/6faf2f50115657241a2d213d29d48c376f8d6996))
- rename expandable vec to threaded vec ([d6d3e0b](https://github.com/saghen/frizbee/commit/d6d3e0bc9052e9c7cc8ef60b199ed1a080631882))

### Documentation

- add note for downloading benchmark data ([aca1794](https://github.com/saghen/frizbee/commit/aca1794c1eb3859e891a37347a92b7ab40b0abe9))
- add basic algorithm description ([e3325e9](https://github.com/saghen/frizbee/commit/e3325e94bba702c2e97490d96e38acdbc2450707))
- add note about typo resistance ([005e443](https://github.com/saghen/frizbee/commit/005e4438b21a8c1d87967cae8eda4774729d0589))
- add benchmarks ([edc713a](https://github.com/saghen/frizbee/commit/edc713aa6773f5e0dbfb5766d4d7af4d03ec83e1))
- benchmarks formatting ([6fc46e9](https://github.com/saghen/frizbee/commit/6fc46e9aa9f63a8211bac65dc71a29c6e20eafc0))
- update benchmarks ([6ce6b70](https://github.com/saghen/frizbee/commit/6ce6b70fe3500576a30758dd5f026a13f67e90e2))
- clarify how 2+ typos works ([90a276c](https://github.com/saghen/frizbee/commit/90a276c41f64e65ba936df0638486240a5cbd3e1))
- cleanup readme ([48d2f36](https://github.com/saghen/frizbee/commit/48d2f36223dcb781772fd1a79e4ae078f05e9e97))
- fix typo ([a535bd7](https://github.com/saghen/frizbee/commit/a535bd75abc6bfe76f532dbb5c374ea0a38e88fd))
- add description and license to Cargo.toml ([6421984](https://github.com/saghen/frizbee/commit/642198445fb8ffde88e4e0b68bb557c47f1a9292))
- add ideas section ([5bf7e6f](https://github.com/saghen/frizbee/commit/5bf7e6f7c5517da54275694a1c8cd89b4390a1bc))
- clarify algorithm ([d490de7](https://github.com/saghen/frizbee/commit/d490de7138c62062a84e9243112ef094f261056f))
- update benchmarks, elaborate on implementation ([f232a0c](https://github.com/saghen/frizbee/commit/f232a0cd7694fa6ba75b055f4e8a8c20529fca88))
- fix indentation in implementation section ([1448e19](https://github.com/saghen/frizbee/commit/1448e19592d789bbd18abad37433c6872f1e9915))
- benchmarks grammar ([bd63226](https://github.com/saghen/frizbee/commit/bd63226acee48d4d30b8b9321c6b03e9bb08faf8))
- add repository to Cargo.toml ([021d2eb](https://github.com/saghen/frizbee/commit/021d2eba64b3981238b14824d05de3432928424c))
- clarify long list tip ([6b65d87](https://github.com/saghen/frizbee/commit/6b65d87e70d881f8e600f8a702a2af8edec2a7db))
- update benchmarks (~1.25x speed-up) ([3150714](https://github.com/saghen/frizbee/commit/3150714c39e18ef9f936d868ddcf04ab4f6e3055))
- fix out of date match_* docs ([ca69df5](https://github.com/saghen/frizbee/commit/ca69df56b84d3f96df0d3cb348af4069a4997bc1))
- drop emojis from benchmarks ([a026b01](https://github.com/saghen/frizbee/commit/a026b01907224b4acf6ace024e26f2e18b6b3d84))

### Performance

- *(typos)* compute max score and position idx using simd (#11) ([f2d0797](https://github.com/saghen/frizbee/commit/f2d0797313bd567efe4e2f831bc335bab83602d8))
- make EXACT_MATCH_BONUS a u16 ([5c7ef42](https://github.com/saghen/frizbee/commit/5c7ef42bc9bd4c5255013dc5b42036de3c5a6529))
- remove unneeded arrays ([825467f](https://github.com/saghen/frizbee/commit/825467fcece1bdce28e4ef8112b4a2fa164dcd61))
- simplify haystack -> simd ([116d061](https://github.com/saghen/frizbee/commit/116d0619f93ea1a99010a39d78a950718e484097))
- move constants to SimdNum trait ([ff9e75b](https://github.com/saghen/frizbee/commit/ff9e75bfdce08465af88628183140c8bd67667bf))
- prefix_mask doesn't need to be simd ([610f490](https://github.com/saghen/frizbee/commit/610f490231d11bbc7a45ca293f6d94d0bf1613b1))
- map needle slice directly, convert to N type in loop ([917f1da](https://github.com/saghen/frizbee/commit/917f1da1a6a73382d288f39f15ad88cf53a4b6d2))
- left_gap_penalty_masks doesn't need to be an array ([00be248](https://github.com/saghen/frizbee/commit/00be2489d1b86871dfbf23a42f745f021272ffb6))

## New Contributors ❤︎

- @saghen made their first contribution
- @dmtrKovalenko made their first contribution in [#47](https://github.com/saghen/frizbee/pull/47)
- @stefanboca made their first contribution
- @ruslanSorokin made their first contribution in [#14](https://github.com/saghen/frizbee/pull/14)
- @Danielkonge made their first contribution in [#3](https://github.com/saghen/frizbee/pull/3)
- @0xJWLabs made their first contribution in [#1](https://github.com/saghen/frizbee/pull/1)

