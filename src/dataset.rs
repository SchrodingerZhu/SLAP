use std::{cell::UnsafeCell, io::Write, str::FromStr};

use burn_dataset::SqliteDatasetWriter;
use indicatif::ParallelProgressIterator;
use rand::prelude::Distribution;
use rayon::iter::ParallelIterator;
use regex_automata::meta::Regex;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::{graph, simulator, Context};

#[derive(Debug, Deserialize, Serialize)]
pub struct Replacement {
    pattern: String,
    lower_bound: isize,
    upper_bound: isize,
    step: isize,
}

impl FromStr for Replacement {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.strip_prefix("<[").ok_or("missing prefix")?;
        let mut iter = s.split(':');
        let pattern = iter
            .next()
            .ok_or("missing symbol")?
            .strip_suffix("]>")
            .ok_or("missing suffix")?
            .to_string();
        let lower_bound = iter
            .next()
            .ok_or("missing lower bound")?
            .parse()
            .ok()
            .ok_or("invalid lower bound")?;
        let upper_bound = iter
            .next()
            .ok_or("missing upper bound")?
            .parse()
            .ok()
            .ok_or("invalid upper bound")?;
        let step = iter
            .next()
            .ok_or("missing step")?
            .parse()
            .ok()
            .ok_or("invalid step")?;
        Ok(Self {
            pattern,
            lower_bound,
            upper_bound,
            step,
        })
    }
}

fn replace(mut s: &str, replacements: &[(&str, usize)]) -> String {
    let map = replacements
        .iter()
        .enumerate()
        .map(|(idx, (_, val))| (idx, val.to_string()))
        .collect::<FxHashMap<_, _>>();
    let regex = Regex::new_many(
        replacements
            .iter()
            .map(|(r, _)| r)
            .collect::<Vec<_>>()
            .as_slice(),
    )
    .unwrap();
    let mut result = String::new();
    while let Some(i) = regex.find(s) {
        let pat = i.pattern();
        let start = i.start();
        let end = i.end();
        result.push_str(&s[..start]);
        result.push_str(map.get(&pat.as_usize()).unwrap());
        s = &s[end..];
        if s.is_empty() {
            break;
        }
    }
    result.push_str(s);
    result
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NetItem {
    pub data: Box<[f32]>,
    pub target: Box<[f32]>,
}

fn simulate_code(input: &str, block_size: usize, cache_size: usize) -> Box<[f32]> {
    let ctx = Context {
        arena: bumpalo::Bump::new(),
        dump_node: false,
        printer: UnsafeCell::new(
            Box::new(std::fs::File::open("/dev/null").unwrap()) as Box<dyn std::io::Write>
        ),
    };
    let mut tempfile = tempfile::NamedTempFile::new().unwrap();
    writeln!(tempfile, "{}", input).unwrap();
    tempfile.flush().unwrap();
    let (g, vaddrs) = graph::Graph::new_from_file(&ctx, &format!("{}", tempfile.path().display()))
        .expect("failed to parse mlir");
    unsafe {
        let mut sctx = simulator::SimulationCtx::new(&ctx, block_size, cache_size, vaddrs);
        sctx.populate_node_info(g);
        let cell = std::cell::UnsafeCell::new(sctx);
        simulator::slap_run_simulation(&cell, g);
        (*cell.get())
            .node_info
            .iter()
            .map(|x| {
                let (acc, cnt) = x
                    .iter()
                    .map(|(a, b)| (*a, *b))
                    .fold((0, 0), |(acc, cnt), (c, d)| (acc + c * d, cnt + d));
                if cnt == 0 {
                    0.0
                } else {
                    acc as f32 / cnt as f32
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }
}

fn generate_all_possible_replacements(set: &[Replacement]) -> Vec<Vec<(&str, usize)>> {
    let mut result = vec![];
    let mut current = vec![];
    fn dfs<'a>(
        set: &'a [Replacement],
        current: &mut Vec<(&'a str, usize)>,
        result: &mut Vec<Vec<(&'a str, usize)>>,
    ) {
        if set.is_empty() {
            result.push(current.clone());
            return;
        }
        let r = &set[0];
        for i in (r.lower_bound..r.upper_bound).step_by(r.step as usize) {
            current.push((&r.pattern, i as usize));
            dfs(&set[1..], current, result);
            current.pop();
        }
    }
    dfs(set, &mut current, &mut result);
    result
}

fn generate_all_code<'a>(
    template: &'a str,
    replacements: &'a [Replacement],
) -> impl ParallelIterator<Item = (String, Vec<usize>)> + use<'a> {
    use rayon::prelude::*;
    let replacements = generate_all_possible_replacements(replacements);
    replacements.into_par_iter().progress().map(|r| {
        let values = r.iter().map(|(_, val)| *val).collect::<Vec<_>>();
        let replaced = replace(template, &r);
        (replaced, values)
    })
}

pub fn generate_dataset(
    template: &str,
    replacements: &[Replacement],
    block_size: usize,
    cache_size: usize,
    writer: &mut SqliteDatasetWriter<NetItem>,
) {
    use rayon::prelude::*;
    generate_all_code(template, replacements)
        .map(|(replaced, values)| {
            let values = values.iter().map(|x| *x as f32).collect::<Vec<_>>();
            let result = simulate_code(&replaced, block_size, cache_size);
            NetItem {
                data: values.into_boxed_slice(),
                target: result,
            }
        })
        .for_each(|item| {
            let mut rng = rand::thread_rng();
            let is_test = rand::distributions::Uniform::new(0, 10).sample(&mut rng) < 2;
            if is_test {
                writer.write("test", &item).unwrap();
            } else {
                writer.write("train", &item).unwrap();
            }
        });

    writer.set_completed().unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replacement() {
        let r: Replacement = "<[i]>:0:10:1".parse().unwrap();
        assert_eq!(r.pattern.as_str(), "i");
        assert_eq!(r.lower_bound, 0);
        assert_eq!(r.upper_bound, 10);
        assert_eq!(r.step, 1);
    }

    #[test]
    fn test_replace() {
        let s = "i + j + k + p";
        let replacements = vec![("i", 1), ("j", 2), ("k", 3)];
        assert_eq!(replace(s, &replacements), "1 + 2 + 3 + p");
    }

    #[test]
    fn test_generate_all_possible_replacements() {
        let set = vec![
            Replacement {
                pattern: "i".to_string(),
                lower_bound: 0,
                upper_bound: 2,
                step: 1,
            },
            Replacement {
                pattern: "j".to_string(),
                lower_bound: 0,
                upper_bound: 2,
                step: 1,
            },
        ];
        let result = generate_all_possible_replacements(&set);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], vec![("i", 0), ("j", 0)]);
        assert_eq!(result[1], vec![("i", 0), ("j", 1)]);
        assert_eq!(result[2], vec![("i", 1), ("j", 0)]);
        assert_eq!(result[3], vec![("i", 1), ("j", 1)]);
    }

    #[test]
    fn test_simulate_code() {
        crate::init();
        const DATA: &str = include_str!("../example/gemm.mlir");
        let result = simulate_code(DATA, 64, 1024);
        assert_eq!(result.as_ref(), [625.0, 625.0, 625.0, 0.0]);
    }

    #[test]
    fn test_generate_all_code() {
        let replacements = vec![
            Replacement {
                pattern: r"\[\:M\:\]".to_string(),
                lower_bound: 10,
                upper_bound: 100,
                step: 10,
            },
            Replacement {
                pattern: r"\[\:N\:\]".to_string(),
                lower_bound: 10,
                upper_bound: 100,
                step: 10,
            },
            Replacement {
                pattern: r"\[\:K\:\]".to_string(),
                lower_bound: 10,
                upper_bound: 100,
                step: 10,
            },
        ];
        const TEMPLATE: &str = include_str!("../example/gemm-template.mlir");
        let result = generate_all_code(TEMPLATE, &replacements).collect::<Vec<_>>();
        for (replaced, values) in result.iter() {
            println!("{} {:?}", replaced, values);
        }
        assert_eq!(result.len(), 1000);
    }

    #[test]
    fn test_generate_dataset() {
        crate::init();
        let replacements = vec![
            Replacement {
                pattern: r"\[\:M\:\]".to_string(),
                lower_bound: 10,
                upper_bound: 100,
                step: 10,
            },
            Replacement {
                pattern: r"\[\:N\:\]".to_string(),
                lower_bound: 10,
                upper_bound: 100,
                step: 10,
            },
            Replacement {
                pattern: r"\[\:K\:\]".to_string(),
                lower_bound: 10,
                upper_bound: 100,
                step: 10,
            },
        ];
        const TEMPLATE: &str = include_str!("../example/gemm-template.mlir");
        let mut writer = SqliteDatasetWriter::new("/tmp/test.db", true).unwrap();
        generate_dataset(TEMPLATE, &replacements, 64, 64, &mut writer);
    }
}
