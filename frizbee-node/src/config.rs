#[derive(Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PartialConfig {
    pub max_typos: Option<u16>,
    pub sort: Option<bool>,
    pub scoring: Option<PartialScoring>,
}

impl From<PartialConfig> for frizbee::Config {
    fn from(partial: PartialConfig) -> Self {
        let default_config = frizbee::Config::default();
        frizbee::Config {
            max_typos: Some(partial.max_typos.unwrap_or(0)),
            sort: partial.sort.unwrap_or(default_config.sort),
            scoring: partial
                .scoring
                .map(Into::into)
                .unwrap_or(default_config.scoring),
        }
    }
}

#[derive(Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PartialScoring {
    pub match_score: Option<u16>,
    pub mismatch_penalty: Option<u16>,
    pub gap_open_penalty: Option<u16>,
    pub gap_extend_penalty: Option<u16>,
    pub prefix_bonus: Option<u16>,
    pub capitalization_bonus: Option<u16>,
    pub matching_case_bonus: Option<u16>,
    pub exact_match_bonus: Option<u16>,
    pub delimiter_bonus: Option<u16>,
}

impl From<PartialScoring> for frizbee::Scoring {
    fn from(partial: PartialScoring) -> Self {
        let default_scoring = frizbee::Scoring::default();
        frizbee::Scoring {
            match_score: partial.match_score.unwrap_or(default_scoring.match_score),
            mismatch_penalty: partial
                .mismatch_penalty
                .unwrap_or(default_scoring.mismatch_penalty),
            gap_open_penalty: partial
                .gap_open_penalty
                .unwrap_or(default_scoring.gap_open_penalty),
            gap_extend_penalty: partial
                .gap_extend_penalty
                .unwrap_or(default_scoring.gap_extend_penalty),
            prefix_bonus: partial.prefix_bonus.unwrap_or(default_scoring.prefix_bonus),
            capitalization_bonus: partial
                .capitalization_bonus
                .unwrap_or(default_scoring.capitalization_bonus),
            matching_case_bonus: partial
                .matching_case_bonus
                .unwrap_or(default_scoring.matching_case_bonus),
            exact_match_bonus: partial
                .exact_match_bonus
                .unwrap_or(default_scoring.exact_match_bonus),
            delimiter_bonus: partial
                .delimiter_bonus
                .unwrap_or(default_scoring.delimiter_bonus),
        }
    }
}
