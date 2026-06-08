/**
 * Application-wide constants and configuration
 */
export const config = {
  // Base URL
  baseUrl: "https://PranavKalkunte.github.io/subvocal",

  // GitHub
  github: {
    repoUrl: "https://github.com/PranavKalkunte/subvocal",
    starsFormatted: {
      compact: "1.2K",
      full: "1,200",
    },
  },

  // Social links
  social: {
    twitter: "https://x.com/PranavKalkunte",
    discord: "https://discord.gg/subvocal",
  },

  // Static stats (used on landing page)
  stats: {
    contributors: "1",
    commits: "37",
    monthlyUsers: "100",
  },
} as const
