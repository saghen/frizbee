mod matcher;

pub use matcher::match_list;

pub trait Appendable<T> {
    fn append(&mut self, value: T);
}

impl<T> Appendable<T> for Vec<T> {
    fn append(&mut self, value: T) {
        self.push(value);
    }
}
