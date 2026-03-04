use neon::types::extract::Json;

mod config;
use config::PartialConfig as Config;

#[derive(Clone)]
pub struct Matcher {
    inner: frizbee::Matcher,
}

#[neon::export]
fn match_list(
    needle: String,
    haystacks: Json<Vec<String>>,
    config: Option<Json<Config>>,
) -> Json<Vec<frizbee::Match>> {
    Json(frizbee::match_list(
        &needle,
        &haystacks.0,
        &config.map(|c| c.0.into()).unwrap_or_default(),
    ))
}

#[neon::export]
fn match_list_indices(
    needle: String,
    haystacks: Json<Vec<String>>,
    config: Option<Json<Config>>,
) -> Json<Vec<frizbee::MatchIndices>> {
    Json(frizbee::match_list_indices(
        &needle,
        &haystacks.0,
        &config.map(|c| c.0.into()).unwrap_or_default(),
    ))
}

#[neon::export(class)]
impl Matcher {
    pub fn new(needle: String, config: Option<Json<Config>>) -> Self {
        Self {
            inner: frizbee::Matcher::new(
                needle.as_str(),
                &config.map(|c| c.0.into()).unwrap_or_default(),
            ),
        }
    }

    fn set_needle(&mut self, needle: String) {
        self.inner.set_needle(needle.as_str());
    }

    fn set_config(&mut self, config: Json<Config>) {
        self.inner.set_config(&config.0.into());
    }

    #[neon(json)]
    fn match_list(&mut self, haystacks: Vec<String>) -> Vec<frizbee::Match> {
        self.inner.match_list(&haystacks)
    }

    #[neon(json)]
    fn match_list_indices(&mut self, haystacks: Vec<String>) -> Vec<frizbee::MatchIndices> {
        self.inner.match_list_indices(&haystacks)
    }
}
