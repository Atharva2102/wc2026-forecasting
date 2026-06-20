export const TEAM_FLAG: Record<string, string> = {
  Algeria: 'dz',
  Argentina: 'ar',
  Australia: 'au',
  Austria: 'at',
  Belgium: 'be',
  'Bosnia and Herzegovina': 'ba',
  Brazil: 'br',
  'Cabo Verde': 'cv',
  Canada: 'ca',
  Colombia: 'co',
  Croatia: 'hr',
  Curaçao: 'cw',
  Czechia: 'cz',
  "Côte d'Ivoire": 'ci',
  'DR Congo': 'cd',
  Ecuador: 'ec',
  Egypt: 'eg',
  England: 'gb-eng',
  France: 'fr',
  Germany: 'de',
  Ghana: 'gh',
  Haiti: 'ht',
  Iran: 'ir',
  Iraq: 'iq',
  Japan: 'jp',
  Jordan: 'jo',
  Mexico: 'mx',
  Morocco: 'ma',
  Netherlands: 'nl',
  'New Zealand': 'nz',
  Norway: 'no',
  Panama: 'pa',
  Paraguay: 'py',
  Portugal: 'pt',
  Qatar: 'qa',
  'Saudi Arabia': 'sa',
  Scotland: 'gb-sct',
  Senegal: 'sn',
  'South Africa': 'za',
  'South Korea': 'kr',
  Spain: 'es',
  Sweden: 'se',
  Switzerland: 'ch',
  Tunisia: 'tn',
  Türkiye: 'tr',
  USA: 'us',
  Uruguay: 'uy',
  Uzbekistan: 'uz',
};

const SUBDIVISION_FLAGS: Record<string, string> = {
  'gb-eng': '🏴',
  'gb-sct': '🏴',
};

export function flagForTeam(team: string): string {
  const code = TEAM_FLAG[team];
  if (!code) {
    console.warn(`No flag mapping found for team: ${team}`);
    return '□';
  }
  if (SUBDIVISION_FLAGS[code]) {
    return SUBDIVISION_FLAGS[code];
  }
  return isoToEmoji(code);
}

function isoToEmoji(code: string): string {
  return code
    .toUpperCase()
    .replace(/./g, (char) =>
      String.fromCodePoint(127397 + char.charCodeAt(0)),
    );
}
