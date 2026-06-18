fn main() {
    bolero::check!()
        .with_iterations(8192)
        .with_max_len(2048)
        .for_each(|input: &[u8]| frizbee::fuzz_support::assert_prefilter(input));
}
