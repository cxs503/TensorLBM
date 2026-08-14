import { api, request } from './request'

// ============================================================================
// 报告模块 API 封装（对接后端 /api/reports）
// ============================================================================

/** 工程 KPI 指标。 */
export interface EngineeringKpis {
  diagnostic_snapshots: number
  force_rows: number
  image_count: number
  runtime_seconds: number | null
  latest_step: number | null
  mean_cd_last: number | null
  std_cd_last: number | null
  mean_cl_last: number | null
  std_cl_last: number | null
  steady_state_score: number | null
  steady_state_detected: boolean
}

/** 单个作业的报告摘要。 */
export interface ReportSummary {
  job_id: string
  name: string
  job_type: string
  status: string
  created_at: string
  started_at: string | null
  completed_at: string | null
  error: string | null
  diagnostic_steps: number
  force_rows: number
  image_count: number
  run_metadata_available: boolean
  report_url: string
  engineering_kpis: EngineeringKpis
}

/** 对比接口中的一行（在摘要基础上附加扁平化指标）。 */
export interface CompareRow extends ReportSummary {
  compare_metrics: Record<string, number>
}

/** KPI 对比接口响应。 */
export interface CompareResponse {
  count: number
  rows: CompareRow[]
  missing: string[]
  metric_summary: Record<string, {
    min: number
    max: number
    mean: number
    best_job_id: string
    best_value: number
  }>
}

/** 获取指定作业的 HTML 工程报告（原文）。 */
export function getReportHtml(jobId: string): Promise<string> {
  return request<string>({
    method: 'get',
    url: `/reports/${jobId}`,
    responseType: 'text',
  })
}

/** 获取指定作业的报告摘要。 */
export function getReportSummary(jobId: string): Promise<ReportSummary> {
  return api.get<ReportSummary>(`/reports/${jobId}/summary`)
}

/**
 * 对比多个作业的 KPI。
 * 后端要求 `ids` 以重复查询参数形式传递（?ids=a&ids=b），
 * 这里手动拼接以避开 axios 默认的 `ids[]=` 数组序列化。
 */
export function compareReportsKpis(ids: string[]): Promise<CompareResponse> {
  const qs = ids.map((id) => `ids=${encodeURIComponent(id)}`).join('&')
  return api.get<CompareResponse>(`/reports/compare/kpis?${qs}`)
}
