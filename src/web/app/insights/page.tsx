import { InsightsClient } from "@/components/dashboard/insights-client";
import { getDashboardData } from "@/lib/data";

export default function InsightsPage() {
  const data = getDashboardData();
  return <InsightsClient data={data} />;
}
