export function qualityBadgeColor(quality: string): string {
  switch (quality) {
    case "I":
      return "bg-emerald-500/90";
    case "II":
      return "bg-sky-500/90";
    case "III":
      return "bg-amber-500/90";
    case "IV":
      return "bg-orange-500/90";
    default:
      return "bg-rose-500/90";
  }
}

export function qualityLabel(quality: string): string {
  return `${quality || "-"} 类`;
}
