fn main() {
    bolero::check!()
        .with_iterations(256)
        .with_max_len(4096)
        .for_each(|input: &[u8]| frizbee::fuzz_support::assert_public_api(input));
}
