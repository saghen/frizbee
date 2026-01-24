#[inline(always)]
pub fn match_haystack(needle: &[(u8, u8)], haystack: &[u8]) -> bool {
    let mut haystack_idx = 0;
    for needle in needle.iter() {
        loop {
            if haystack_idx == haystack.len() {
                return false;
            }

            if needle.0 == haystack[haystack_idx] || needle.1 == haystack[haystack_idx] {
                haystack_idx += 1;
                break;
            }
            haystack_idx += 1;
        }
    }

    true
}

#[inline(always)]
pub fn match_haystack_typos(needle: &[(u8, u8)], haystack: &[u8], max_typos: u16) -> bool {
    let mut haystack_idx = 0;
    let mut typos = 0;
    for needle in needle.iter() {
        loop {
            if haystack_idx == haystack.len() {
                typos += 1;
                if typos > max_typos as usize {
                    return false;
                }

                haystack_idx = 0;
                break;
            }

            if needle.0 == haystack[haystack_idx] || needle.1 == haystack[haystack_idx] {
                haystack_idx += 1;
                break;
            }
            haystack_idx += 1;
        }
    }

    true
}
