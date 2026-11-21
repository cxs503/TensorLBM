import { api } from './request'

// ============================================================================
// 数据生产模块 API 封装（对接后端 /api/jobs、/api/cad、/api/preprocess、
// /api/solve、/api/postprocess、/api/benchmarks）
// ============================================================================

// ---------------------------------------------------------------------------
// 通用作业（Job）类型
// ---------------------------------------------------------------------------

export interface Job {
  job_id: string
  name: string
  job_type: string
  config: Record<string, any>
  status: string // queued | running | completed | failed | cancelled
  created_at: string
  started_at: string | null
  completed_at: string | null
  error: string | null
  output_dir: string
  logs: string[]
  diagnostics: Record<string, any>[]
  result: Record<string, any>
  cancel_requested: boolean
  queue_wait_seconds: number | null
  run_duration_seconds: number | null
  total_duration_seconds: number | null
  io_bytes: number
  retry_attempt: number
  max_retries: number
  failure_category: string | null
  assigned_resource: string
  estimated_cost: number
  priority: number
  convergence_status: string
}

export interface JobListResponse {
  jobs: Job[]
  total: number
  offset: number
  limit: number
}

export function listJobs(params?: {
  status?: string
  limit?: number
  offset?: number
}): Promise<JobListResponse> {
  return api.get<JobListResponse>('/jobs', params)
}

export function getJob(jobId: string): Promise<Job> {
  return api.get<Job>(`/jobs/${jobId}`)
}

export function cancelJob(jobId: string): Promise<{ job_id: string; status: string }> {
  return api.post<{ job_id: string; status: string }>(`/jobs/${jobId}/cancel`)
}

export function deleteJob(jobId: string): Promise<{ deleted: string }> {
  return api.delete<{ deleted: string }>(`/jobs/${jobId}`)
}

export function getJobLogs(jobId: string): Promise<{ job_id: string; logs: string[] }> {
  return api.get<{ job_id: string; logs: string[] }>(`/jobs/${jobId}/logs`)
}

export interface JobFile {
  path: string
  size: number
  mime: string
}

export function getJobFiles(jobId: string): Promise<{ job_id: string; files: JobFile[] }> {
  return api.get<{ job_id: string; files: JobFile[] }>(`/jobs/${jobId}/files`)
}

export function getJobImages(jobId: string): Promise<{ job_id: string; images: string[] }> {
  return api.get<{ job_id: string; images: string[] }>(`/jobs/${jobId}/images`)
}

export function getJobMetadata(jobId: string): Promise<{ job_id: string; metadata: Record<string, any> }> {
  return api.get<{ job_id: string; metadata: Record<string, any> }>(`/jobs/${jobId}/metadata`)
}

export interface LiveMetricsResponse {
  job_id: string
  status: string
  total_diagnostics: number
  diagnostics: Record<string, any>[]
  has_more: boolean
}

export function getLiveMetrics(jobId: string, sinceStep = 0): Promise<LiveMetricsResponse> {
  return api.get<LiveMetricsResponse>(`/jobs/${jobId}/live-metrics`, { since_step: sinceStep })
}

/**
 * 构造作业输出文件的静态下载 URL（用于 <img> / <a download>）。
 * 后端通过 /api/jobs/{job_id}/files/{path} 直接返回文件内容。
 */
export function jobFileUrl(jobId: string, filePath: string): string {
  const safePath = filePath
    .split('/')
    .map((seg) => encodeURIComponent(seg))
    .join('/')
  return `/api/jobs/${jobId}/files/${safePath}`
}

// ---------------------------------------------------------------------------
// CAD 建模
// ---------------------------------------------------------------------------

export interface HullTypeItem {
  value: string
  label: string
  description: string
  Cb: number
}

export function cadHullTypes(): Promise<{ hull_types: HullTypeItem[] }> {
  return api.get<{ hull_types: HullTypeItem[] }>('/cad/hull-types')
}

export interface CadPreviewResult {
  image: string
  stats: Record<string, any>
}

export function cadPreview(data: {
  hull_type: string
  length: number
  beam: number
  draft: number
  n_stations: number
}): Promise<CadPreviewResult> {
  return api.post<CadPreviewResult>('/cad/preview', data)
}

export function cadHullMask(data: {
  hull_type: string
  nx: number
  ny: number
  nz: number
  length: number
  beam: number
  draft: number
  device?: string
}): Promise<{ image: string; stats: Record<string, any> }> {
  return api.post<{ image: string; stats: Record<string, any> }>('/cad/hull-mask', data)
}

export function cadLbmParameters(data: {
  length_m: number
  speed_ms: number
  nu_m2s: number
  lbm_length: number
  lbm_speed: number
  froude_target: number | null
}): Promise<Record<string, any>> {
  return api.post<Record<string, any>>('/cad/lbm-parameters', data)
}

export function cadSendToSolver(data: Record<string, any>): Promise<{ job_id: string; message: string }> {
  return api.post<{ job_id: string; message: string }>('/cad/send-to-solver', data)
}

export function cadResistanceEstimate(data: Record<string, any>): Promise<Record<string, any>> {
  return api.post<Record<string, any>>('/cad/resistance-estimate', data)
}

// ---------------------------------------------------------------------------
// 预处理
// ---------------------------------------------------------------------------

export function preprocessPolygonMask(data: {
  nx: number
  ny: number
  vertices: number[][]
}): Promise<{
  nx: number
  ny: number
  obstacle_cells: number
  fluid_cells: number
  image: string
}> {
  return api.post('/preprocess/polygon-mask', data)
}

export function preprocessRandomPorosity(data: {
  nx: number
  ny: number
  porosity: number
  sigma: number
  seed: number
}): Promise<{
  nx: number
  ny: number
  requested_porosity: number
  actual_porosity: number
  image: string
}> {
  return api.post('/preprocess/random-porosity-2d', data)
}

export function preprocessUnits(data: {
  phys_length_m: number
  phys_velocity_ms: number
  phys_nu_m2s: number
  lbm_length: number
  lbm_velocity: number
}): Promise<Record<string, any>> {
  return api.post<Record<string, any>>('/preprocess/units', data)
}

export function preprocessYPlus(data: {
  re: number
  u_ms: number
  l_m: number
  nu_m2s: number
  target_yplus: number
  n_cells: number
  geometry: string
}): Promise<Record<string, any>> {
  return api.post<Record<string, any>>('/preprocess/yplus', data)
}

export interface Material {
  id: string
  name: string
  name_zh: string
  category: string
  density_kg_m3: number
  dynamic_viscosity_pa_s: number
  kinematic_viscosity_m2_s: number
  ref_temp_c: number
}

export function preprocessMaterials(category?: string): Promise<{ count: number; materials: Material[] }> {
  return api.get<{ count: number; materials: Material[] }>('/preprocess/materials', category ? { category } : undefined)
}

export function preprocessPreflight(data: Record<string, any>): Promise<Record<string, any>> {
  return api.post<Record<string, any>>('/preprocess/preflight', data)
}

// ---------------------------------------------------------------------------
// 求解器
// ---------------------------------------------------------------------------

export function submitSolverJob(
  endpoint: string,
  data: Record<string, any>,
): Promise<{ job_id: string; message: string }> {
  return api.post<{ job_id: string; message: string }>(endpoint, data)
}

export function solverValidate(data: Record<string, any>): Promise<Record<string, any>> {
  return api.post<Record<string, any>>('/solve/validate', data)
}

// ---------------------------------------------------------------------------
// 后处理
// ---------------------------------------------------------------------------

export function postSummary(jobId: string): Promise<Record<string, any>> {
  return api.get<Record<string, any>>(`/postprocess/summary/${jobId}`)
}

export function postCheckpoints(jobId: string): Promise<{ job_id: string; checkpoints: string[] }> {
  return api.get<{ job_id: string; checkpoints: string[] }>(`/postprocess/checkpoints/${jobId}`)
}

export interface FieldDataResponse {
  job_id: string
  step: number
  field: string
  nx: number
  ny: number
  nx_orig: number
  ny_orig: number
  field_min: number
  field_max: number
  data: number[]
  ux: number[]
  uy: number[]
}

export function postFieldData(
  jobId: string,
  params: { field: string; checkpoint: string },
): Promise<FieldDataResponse> {
  return api.get<FieldDataResponse>(`/postprocess/field-data/${jobId}`, params)
}

export function postConvergence(jobId: string): Promise<Record<string, any>> {
  return api.get<Record<string, any>>(`/postprocess/convergence/${jobId}`)
}

export function postVelocityProfile(data: {
  job_id: string
  direction: string
  position: number
}): Promise<Record<string, any>> {
  return api.post<Record<string, any>>('/postprocess/velocity-profile', data)
}

export function postSnapshotAnalysis(jobId: string): Promise<Record<string, any>> {
  return api.get<Record<string, any>>(`/postprocess/snapshot-analysis/${jobId}`)
}

// ---------------------------------------------------------------------------
// 基准
// ---------------------------------------------------------------------------

export function submitBenchmark(
  endpoint: string,
  data: Record<string, any>,
): Promise<{ job_id: string; message: string }> {
  return api.post<{ job_id: string; message: string }>(endpoint, data)
}

export function benchmarkAccuracyBaselines(): Promise<Record<string, any>> {
  return api.get<Record<string, any>>('/benchmarks/accuracy/baselines')
}

export function benchmarkAccuracyReport(jobId: string): Promise<Record<string, any>> {
  return api.get<Record<string, any>>(`/benchmarks/accuracy/report/${jobId}`)
}

export function benchmarkAcceptanceGates(): Promise<Record<string, any>> {
  return api.get<Record<string, any>>('/benchmarks/acceptance-gates')
}
