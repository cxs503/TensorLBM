import { api } from './request'

// ---------------------------------------------------------------------------
// Types (mirror app/backend/schemas/apps.py)
// ---------------------------------------------------------------------------

/** 一个已注册 AI4S 应用的标识信息。 */
export interface AppInfo {
  name: string
  family: string
  version: string
}

/** GET /api/apps 返回的应用列表信封。 */
export interface AppListResponse {
  apps: AppInfo[]
  total: number
}

/** 可选的 HPC 派发参数。 */
export interface HpcRequest {
  partition?: string
  nodes?: number
  cpus?: number
  mem?: string
  walltime?: string
  backend?: string
}

/** POST /api/apps/{name}/run 的请求体。 */
export interface AppRunRequest {
  produce_cfg?: Record<string, unknown>
  train_cfg?: Record<string, unknown>
  db_path?: string | null
  hpc?: HpcRequest | null
}

/** 本地全栈运行完成后的 RunReport。 */
export interface RunReportOut {
  name: string
  family: string
  data_asset_id: string
  dataset_asset_id: string
  job_id: string
  model_id: number
  metrics: Record<string, unknown>
  lineage_upstream: string[]
}

/** HPC 派发后的提交响应。 */
export interface HpcSubmitResponse {
  app_name: string
  job_id: string
  hpc_job_id: string
  status: string
  backend: string
  script_cmd: string
}

/** 运行应用的响应：本地全栈返回 RunReport，HPC 派发返回 HpcSubmitResponse。 */
export type RunResponse = RunReportOut | HpcSubmitResponse

/** GET /api/apps/{name}/run/{job_id} 的运行状态。 */
export interface RunStatusResponse {
  job_id: string
  app_name: string
  status: string
  hpc_job_id?: string | null
  scheduler_state?: string | null
  elapsed?: string | null
}

// ---------------------------------------------------------------------------
// Type guards
// ---------------------------------------------------------------------------

/** 判断运行响应是否为 HPC 派发响应（否则为 RunReport）。 */
export function isHpcSubmitResponse(res: RunResponse): res is HpcSubmitResponse {
  return 'hpc_job_id' in res
}

/** 判断运行响应是否为本地全栈 RunReport。 */
export function isRunReport(res: RunResponse): res is RunReportOut {
  return !isHpcSubmitResponse(res)
}

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

/** 列出已注册的 AI4S 应用。 */
export function listApps(): Promise<AppListResponse> {
  return api.get<AppListResponse>('/apps')
}

/** 运行一个应用的 full-stack pipeline（本地全栈 or HPC 派发）。 */
export function runApp(name: string, body: AppRunRequest): Promise<RunResponse> {
  return api.post<RunResponse>(`/apps/${encodeURIComponent(name)}/run`, body)
}

/** 查询一次运行的状态。 */
export function getRunStatus(name: string, jobId: string): Promise<RunStatusResponse> {
  return api.get<RunStatusResponse>(`/apps/${encodeURIComponent(name)}/run/${encodeURIComponent(jobId)}`)
}
