use super::Backend;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Utf8ToUtf32Scalar;

impl Backend for Utf8ToUtf32Scalar {
    #[inline(always)]
    fn is_available() -> bool {
        true
    }

    #[inline(always)]
    unsafe fn convert_into(input: &str, out: &mut Vec<u32>) {
        out.clear();
        out.reserve(input.len());
        out.extend(input.chars().map(|c| c as u32));
    }
}
