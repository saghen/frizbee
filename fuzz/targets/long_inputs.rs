fn main() {
    bolero::check!()
        .with_iterations(128)
        .with_max_len(8192)
        .for_each(|input: &[u8]| frizbee::fuzz_support::assert_public_api_long_inputs(input));
}
