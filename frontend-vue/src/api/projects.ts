import { api } from './request'

// ============================================================================
// 项目/案例模块 API 封装（对接后端 /api/projects）
// 项目 → 案例（仿真案例）层级，包含 CRUD、克隆、工作流推进。
// ============================================================================

/** 工作流阶段（与后端 WORKFLOW_STAGES 保持一致）。 */
export const WORKFLOW_STAGES = ['draft', 'setup', 'meshed', 'solved', 'post_processed']

/** 项目实体。 */
export interface Project {
  id: string
  name: string
  description: string
  owner: string
  tags: string[]
  created_at: string
  updated_at: string
}

/** 仿真案例实体。 */
export interface SimulationCase {
  id: string
  project_id: string
  name: string
  description: string
  scenario: string
  status: string
  workflow_stage: string
  config: Record<string, any>
  job_id: string | null
  created_at: string
  updated_at: string
}

export interface ProjectCreatePayload {
  name: string
  description?: string
  owner?: string
  tags?: string[]
}

export interface ProjectUpdatePayload {
  name?: string
  description?: string
  owner?: string
  tags?: string[]
}

export interface CaseCreatePayload {
  name: string
  description?: string
  scenario?: string
  workflow_stage?: string
  config?: Record<string, any>
}

export interface CaseUpdatePayload {
  name?: string
  description?: string
  scenario?: string
  status?: string
  workflow_stage?: string
  config?: Record<string, any>
  job_id?: string | null
}

export interface CloneCasePayload {
  name?: string
  config_overrides?: Record<string, any>
}

// ---------------------------------------------------------------------------
// 项目
// ---------------------------------------------------------------------------

export function listProjects(): Promise<Project[]> {
  return api.get<Project[]>('/projects')
}

export function createProject(data: ProjectCreatePayload): Promise<Project> {
  return api.post<Project>('/projects', data)
}

export function getProject(projectId: string): Promise<Project> {
  return api.get<Project>(`/projects/${projectId}`)
}

export function updateProject(projectId: string, data: ProjectUpdatePayload): Promise<Project> {
  return api.put<Project>(`/projects/${projectId}`, data)
}

export function deleteProject(projectId: string): Promise<void> {
  return api.delete<void>(`/projects/${projectId}`)
}

// ---------------------------------------------------------------------------
// 案例（嵌套在项目下）
// ---------------------------------------------------------------------------

export function listCases(projectId: string): Promise<SimulationCase[]> {
  return api.get<SimulationCase[]>(`/projects/${projectId}/cases`)
}

export function createCase(projectId: string, data: CaseCreatePayload): Promise<SimulationCase> {
  return api.post<SimulationCase>(`/projects/${projectId}/cases`, data)
}

export function updateCase(
  projectId: string,
  caseId: string,
  data: CaseUpdatePayload,
): Promise<SimulationCase> {
  return api.put<SimulationCase>(`/projects/${projectId}/cases/${caseId}`, data)
}

export function deleteCase(projectId: string, caseId: string): Promise<void> {
  return api.delete<void>(`/projects/${projectId}/cases/${caseId}`)
}

export function cloneCase(
  projectId: string,
  caseId: string,
  data: CloneCasePayload = {},
): Promise<SimulationCase> {
  return api.post<SimulationCase>(`/projects/${projectId}/cases/${caseId}/clone`, data)
}

export function advanceWorkflow(projectId: string, caseId: string): Promise<SimulationCase> {
  return api.post<SimulationCase>(`/projects/${projectId}/cases/${caseId}/advance-workflow`)
}
