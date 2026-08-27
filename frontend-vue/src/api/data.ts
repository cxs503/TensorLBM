import { api } from './request'

// ---------------------------------------------------------------------------
// Types (mirror app/backend/schemas/data_catalog.py)
// ---------------------------------------------------------------------------

export interface Asset {
  asset_id: string
  name: string
  kind: string
  description: string
  field_name: string | null
  units: string | null
  shape: string | null
  dtype: string | null
  tags: string[]
  quality_score: number
  sensitivity_level: string
  source_run_id: string | null
  status: string
  version: string
  created_at: number
  updated_at: number
}

export interface AssetListResponse {
  assets: Asset[]
  total: number
  limit: number
  offset: number
}

export interface AssetCreatePayload {
  asset_id: string
  name: string
  kind?: string
  description?: string
  field_name?: string | null
  units?: string | null
  shape?: string | null
  dtype?: string | null
  tags?: string[]
  quality_score?: number
  sensitivity_level?: string
  source_run_id?: string | null
  status?: string
  version?: string
}

export interface AssetUpdatePayload {
  name?: string
  description?: string
  tags?: string[]
  status?: string
  quality_score?: number
}

export interface MetadataEntry {
  key: string
  value: string
  source: string
  confidence: number
}

export interface MetadataCreatePayload {
  key: string
  value: string
  source?: string
  confidence?: number
}

export interface LineageEdge {
  source_id: string
  target_id: string
  relation_type: string
  transformation: string
  resource_type: string
}

export interface LineageResponse {
  asset_id: string
  lineage: LineageEdge[]
  upstream: string[]
}

export interface LineageCreatePayload {
  target_id: string
  relation_type?: string
  transformation?: string
  resource_type?: string
}

export interface QualityCheckItem {
  check_name: string
  passed: boolean
  detail: string
}

export interface QualityCheckRequest {
  asset_id: string
  data: number[][]
  mass_field?: boolean
  mass_tol?: number
}

export interface QualityCheckResponse {
  asset_id: string
  overall_score: number
  status: string
  checks: QualityCheckItem[]
}

export interface QualityReport {
  checks: QualityCheckItem[]
  overall_score: number
  status: string
  created_at: number
}

export interface AssetListQuery {
  kind?: string
  field_name?: string
  status?: string
  limit?: number
  offset?: number
}

// ---------------------------------------------------------------------------
// Asset endpoints
// ---------------------------------------------------------------------------

export function listAssets(params: AssetListQuery = {}) {
  return api.get<AssetListResponse>('/data/assets', params)
}

export function getAsset(assetId: string) {
  return api.get<Asset>(`/data/assets/${assetId}`)
}

export function registerAsset(payload: AssetCreatePayload) {
  return api.post<Asset>('/data/assets', payload)
}

export function updateAsset(assetId: string, payload: AssetUpdatePayload) {
  return api.put<Asset>(`/data/assets/${assetId}`, payload)
}

export function archiveAsset(assetId: string) {
  return api.delete<{ asset_id: string; status: string }>(`/data/assets/${assetId}`)
}

// ---------------------------------------------------------------------------
// Metadata endpoints
// ---------------------------------------------------------------------------

export function listMetadata(assetId: string) {
  return api.get<MetadataEntry[]>(`/data/assets/${assetId}/metadata`)
}

export function addMetadata(assetId: string, payload: MetadataCreatePayload) {
  return api.post<MetadataEntry>(`/data/assets/${assetId}/metadata`, payload)
}

export function deleteMetadata(assetId: string, key: string) {
  return api.delete<{ asset_id: string; key: string; deleted: string }>(
    `/data/assets/${assetId}/metadata`,
    { key },
  )
}

// ---------------------------------------------------------------------------
// Lineage endpoints
// ---------------------------------------------------------------------------

export function getLineage(assetId: string) {
  return api.get<LineageResponse>(`/data/assets/${assetId}/lineage`)
}

export function addLineage(assetId: string, payload: LineageCreatePayload) {
  return api.post<LineageEdge>(`/data/assets/${assetId}/lineage`, payload)
}

// ---------------------------------------------------------------------------
// Quality endpoints
// ---------------------------------------------------------------------------

export function runQualityCheck(payload: QualityCheckRequest) {
  return api.post<QualityCheckResponse>('/data/quality/check', payload)
}

export function listQualityReports(assetId: string, limit = 10) {
  return api.get<QualityReport[]>(`/data/quality/${assetId}/reports`, { limit })
}
