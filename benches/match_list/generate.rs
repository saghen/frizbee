use std::fmt::Display;

use rand::prelude::*;
use rand_distr::{Alphanumeric, Distribution, Normal};

enum MatchType {
    None,
    Partial,
    Full,
}

#[derive(Clone, Debug)]
pub struct HaystackGenerationOptions {
    /// Seed for the random number generator to ensure consistent data.
    pub seed: u64,
    /// Percentage of data that should match partially
    pub partial_match_percentage: f64,
    /// Percentage of data that should match
    pub match_percentage: f64,
    /// Median length of the strings
    pub median_length: usize,
    /// Standard deviation of the string lengths
    pub std_dev_length: usize,
    /// Number of data samples to generate
    pub num_samples: usize,
}

impl Display for HaystackGenerationOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let match_title = if self.match_percentage == 0.0 && self.partial_match_percentage == 0.0 {
            "no_match".to_string()
        } else if self.match_percentage == 1.0 {
            "all_match".to_string()
        } else {
            format!("{:.0}%_match", self.match_percentage * 100.0,)
        };
        write!(
            f,
            "length_{}/{}/items_{}",
            self.median_length, match_title, self.num_samples,
        )
    }
}

/// Generates a dataset matching the specified criteria.
/// NOTE: The length of the generated strings may not match the median if the needle
/// is close to or longer than the median length.
pub fn generate_haystack(needle: &str, options: HaystackGenerationOptions) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(options.seed);

    assert!(options.partial_match_percentage + options.match_percentage <= 1.0);

    // Create a normal distribution for the lengths
    let normal_dist = Normal::new(options.median_length as f64, options.std_dev_length as f64)
        .expect("failed to create normal distribution");

    (0..options.num_samples)
        .map(|_| {
            let mut rng = StdRng::seed_from_u64(rng.random());

            // Decide if this entry should be a match
            let match_type = match rng.random::<f64>() {
                x if x < options.partial_match_percentage => MatchType::Partial,
                x if x < options.partial_match_percentage + options.match_percentage => {
                    MatchType::Full
                }
                _ => MatchType::None,
            };

            // Generate a length from the normal distribution
            let length = normal_dist.sample(&mut rng).round().abs().max(1.) as usize;

            match match_type {
                // Generate a random alphanumeric string of the desired length
                // skipping any characters that are in the needle
                MatchType::None => rng
                    .sample_iter(&Alphanumeric)
                    .map(char::from)
                    .filter(|c| {
                        !needle.contains(c.to_ascii_lowercase())
                            && !needle.contains(c.to_ascii_uppercase())
                    })
                    .take(length)
                    .collect(),

                // Generate a random string of the desired length with a random number of matching
                // characters
                MatchType::Partial => {
                    // Get random characters from the needle
                    let match_count = rng.random_range(0..length.min(needle.len()));
                    let needle_chars = generate_unique_indices(match_count, needle.len(), &mut rng)
                        .into_iter()
                        .map(|i| needle.chars().nth(i).unwrap())
                        .collect::<Vec<char>>();

                    // Get remaining characters to fill the remaining length randomly
                    let remaining_chars = StdRng::seed_from_u64(rng.random())
                        .sample_iter(Alphanumeric)
                        .map(char::from)
                        .filter(|c| {
                            !needle.contains(c.to_ascii_lowercase())
                                && !needle.contains(c.to_ascii_uppercase())
                        })
                        .take(length - match_count)
                        .collect::<Vec<char>>();

                    join_randomly(&needle_chars, &remaining_chars, &mut rng)
                        .iter()
                        .collect()
                }

                // Generate a random string that matches the entire needle, with additional random
                // characters
                // NOTE: The length of the generated string may not match the desired length
                // if the needle is close to or longer than the median length.
                MatchType::Full => {
                    let needle_chars = needle.chars().collect::<Vec<char>>();
                    let remaining_chars = (0..(length.saturating_sub(needle.len())))
                        .map(|_| rng.sample(Alphanumeric).into())
                        .collect::<Vec<char>>();

                    join_randomly(&needle_chars, &remaining_chars, &mut rng)
                        .iter()
                        .collect()
                }
            }
        })
        .collect::<Vec<String>>()
}

/// Generates a vector of unique indices from 0 to `y` with a maximum of `x` unique indices.
fn generate_unique_indices(x: usize, y: usize, rng: &mut StdRng) -> Vec<usize> {
    assert!(
        x <= y,
        "Cannot generate more unique indices than the maximum value"
    );

    let mut indices: Vec<usize> = (0..y).collect();
    // Shuffle the indices to introduce randomness
    indices.shuffle(rng);
    // Take the first `x` elements
    indices.truncate(x);
    // Sort them to ensure they are in increasing order
    indices.sort();

    indices
}

fn join_randomly<T>(a: &[T], b: &[T], rng: &mut StdRng) -> Vec<T>
where
    T: Copy,
{
    // Get the chance of picking an element from `a`
    let pick_chance = (a.len() as f64) / (a.len() + b.len()) as f64;

    // Iterate over the elements and pick randomly
    let mut a_index = 0;
    let mut b_index = 0;
    let mut result = Vec::new();
    while a_index < a.len() && b_index < b.len() {
        if rng.random::<f64>() < pick_chance {
            result.push(a[a_index]);
            a_index += 1;
        } else {
            result.push(b[b_index]);
            b_index += 1;
        }
    }

    // Add remaining elements
    result.extend(a[a_index..].iter().copied());
    result.extend(b[b_index..].iter().copied());

    result
}
