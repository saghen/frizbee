declare module "frizbee" {
  function matchList(
    needle: string,
    haystacks: string[],
    config?: Config,
  ): Match[];

  function matchListIndices(
    needle: string,
    haystacks: string[],
    config?: Config,
  ): MatchIndices[];

  export class Matcher {
    constructor(needle: string, config?: Config);
    setNeedle(needle: string): void;
    setConfig(config: Config): void;
    matchList(haystacks: string[]): Match[];
    matchListIndices(haystacks: string[]): MatchIndices[];
  }

  export interface Config {
    maxTypos?: number;
    sort?: boolean;
    scoring?: Scoring;
  }

  export interface Scoring {
    matchScore?: number;
    mismatchPenalty?: number;
    gapOpenPenalty?: number;
    gapExtendPenalty?: number;
    prefixBonus?: number;
    capitalizationBonus?: number;
    matchingCaseBonus?: number;
    exactMatchBonus?: number;
    delimiterBonus?: number;
  }

  export interface Match {
    index: number;
    score: number;
    exact: boolean;
  }

  export interface MatchIndices {
    score: number;
    index: number;
    exact: boolean;
    indices: number[];
  }
}
