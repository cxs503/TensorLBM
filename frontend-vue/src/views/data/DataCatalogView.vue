<template>
  <el-card shadow="never">
    <!-- 过滤栏 -->
    <div class="toolbar">
      <div class="filters">
        <el-select
          v-model="filters.kind"
          placeholder="资产类型"
          clearable
          style="width: 160px"
          @change="handleSearch"
        >
          <el-option
            v-for="opt in kindOptions"
            :key="opt.value"
            :label="opt.label"
            :value="opt.value"
          />
        </el-select>
        <el-input
          v-model="filters.field_name"
          placeholder="字段名 (field_name)"
          clearable
          style="width: 200px"
          @keyup.enter="handleSearch"
          @clear="handleSearch"
        />
        <el-select
          v-model="filters.status"
          placeholder="状态"
          style="width: 140px"
          @change="handleSearch"
        >
          <el-option
            v-for="opt in statusOptions"
            :key="opt.value"
            :label="opt.label"
            :value="opt.value"
          />
        </el-select>
        <el-button type="primary" :icon="Search" @click="handleSearch">查询</el-button>
        <el-button :icon="Refresh" @click="handleReset">重置</el-button>
      </div>
      <div class="actions">
        <el-button type="success" :icon="Plus" @click="openRegister">登记资产</el-button>
      </div>
    </div>

    <!-- 资产列表 -->
    <el-table
      v-loading="loading"
      :data="assets"
      stripe
      border
      row-key="asset_id"
      @row-click="openDetail"
    >
      <el-table-column prop="asset_id" label="资产 ID" min-width="180" show-overflow-tooltip />
      <el-table-column prop="name" label="名称" min-width="140" show-overflow-tooltip />
      <el-table-column prop="kind" label="类型" width="140">
        <template #default="{ row }">
          <el-tag :type="kindTagType(row.kind)" size="small">{{ row.kind }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="field_name" label="字段名" min-width="120" show-overflow-tooltip>
        <template #default="{ row }">
          <span>{{ row.field_name || '—' }}</span>
        </template>
      </el-table-column>
      <el-table-column prop="quality_score" label="质量分" width="160">
        <template #default="{ row }">
          <el-progress
            :percentage="row.quality_score"
            :color="qualityColor(row.quality_score)"
            :stroke-width="14"
            :text-inside="true"
          />
        </template>
      </el-table-column>
      <el-table-column prop="status" label="状态" width="110">
        <template #default="{ row }">
          <el-tag :type="row.status === 'active' ? 'success' : 'info'" size="small">
            {{ row.status }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="更新时间" width="170">
        <template #default="{ row }">
          <span>{{ formatTime(row.updated_at) }}</span>
        </template>
      </el-table-column>
      <el-table-column label="操作" width="150" fixed="right">
        <template #default="{ row }">
          <el-button link type="primary" size="small" @click.stop="openDetail(row)">详情</el-button>
          <el-button link type="danger" size="small" @click.stop="handleArchive(row)">归档</el-button>
        </template>
      </el-table-column>
    </el-table>

    <!-- 分页 -->
    <div class="pagination">
      <el-pagination
        v-model:current-page="pagination.page"
        v-model:page-size="pagination.pageSize"
        :total="total"
        :page-sizes="[10, 20, 50, 100]"
        layout="total, sizes, prev, pager, next, jumper"
        @current-change="fetchAssets"
        @size-change="handleSizeChange"
      />
    </div>

    <!-- 登记资产对话框 -->
    <el-dialog
      v-model="registerVisible"
      title="登记资产"
      width="640px"
      :close-on-click-modal="false"
    >
      <el-form :model="registerForm" label-width="120px">
        <el-row :gutter="16">
          <el-col :span="12">
            <el-form-item label="资产 ID" required>
              <el-input v-model="registerForm.asset_id" placeholder="全局唯一标识" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="名称" required>
              <el-input v-model="registerForm.name" placeholder="资产名称" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="类型">
              <el-select v-model="registerForm.kind" style="width: 100%">
                <el-option
                  v-for="opt in kindOptions.filter((k) => k.value)"
                  :key="opt.value"
                  :label="opt.label"
                  :value="opt.value"
                />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="字段名">
              <el-input v-model="registerForm.field_name" placeholder="field_name" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="单位">
              <el-input v-model="registerForm.units" placeholder="units" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="数据类型">
              <el-input v-model="registerForm.dtype" placeholder="float32" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="形状">
              <el-input v-model="registerForm.shape" placeholder="[128, 128]" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="版本">
              <el-input v-model="registerForm.version" placeholder="1.0.0" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="来源 run_id">
              <el-input v-model="registerForm.source_run_id" placeholder="source_run_id" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="敏感级别">
              <el-select v-model="registerForm.sensitivity_level" style="width: 100%">
                <el-option label="internal" value="internal" />
                <el-option label="public" value="public" />
                <el-option label="restricted" value="restricted" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="质量分">
              <el-input-number
                v-model="registerForm.quality_score"
                :min="0"
                :max="100"
                style="width: 100%"
              />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="标签">
              <el-input v-model="registerForm.tags" placeholder="逗号分隔" />
            </el-form-item>
          </el-col>
          <el-col :span="24">
            <el-form-item label="描述">
              <el-input v-model="registerForm.description" type="textarea" :rows="2" />
            </el-form-item>
          </el-col>
        </el-row>
      </el-form>
      <template #footer>
        <el-button @click="registerVisible = false">取消</el-button>
        <el-button type="primary" :loading="registerLoading" @click="submitRegister">提交</el-button>
      </template>
    </el-dialog>

    <!-- 详情抽屉 -->
    <el-drawer
      v-model="drawerVisible"
      :title="drawerTitle"
      size="640px"
      destroy-on-close
    >
      <div v-loading="detailLoading" class="detail-body">
        <el-tabs v-model="activeTab">
          <el-tab-pane label="基本信息" name="info">
            <el-descriptions v-if="detailAsset" :column="1" border>
              <el-descriptions-item label="资产 ID">{{ detailAsset.asset_id }}</el-descriptions-item>
              <el-descriptions-item label="名称">{{ detailAsset.name }}</el-descriptions-item>
              <el-descriptions-item label="类型">{{ detailAsset.kind }}</el-descriptions-item>
              <el-descriptions-item label="字段名">{{ detailAsset.field_name || '—' }}</el-descriptions-item>
              <el-descriptions-item label="单位">{{ detailAsset.units || '—' }}</el-descriptions-item>
              <el-descriptions-item label="形状">{{ detailAsset.shape || '—' }}</el-descriptions-item>
              <el-descriptions-item label="数据类型">{{ detailAsset.dtype || '—' }}</el-descriptions-item>
              <el-descriptions-item label="质量分">{{ detailAsset.quality_score }}</el-descriptions-item>
              <el-descriptions-item label="敏感级别">{{ detailAsset.sensitivity_level }}</el-descriptions-item>
              <el-descriptions-item label="来源 run_id">{{ detailAsset.source_run_id || '—' }}</el-descriptions-item>
              <el-descriptions-item label="状态">{{ detailAsset.status }}</el-descriptions-item>
              <el-descriptions-item label="版本">{{ detailAsset.version }}</el-descriptions-item>
              <el-descriptions-item label="标签">
                <el-tag v-for="t in detailAsset.tags" :key="t" size="small" class="tag-gap">
                  {{ t }}
                </el-tag>
                <span v-if="!detailAsset.tags.length">—</span>
              </el-descriptions-item>
              <el-descriptions-item label="描述">{{ detailAsset.description || '—' }}</el-descriptions-item>
              <el-descriptions-item label="创建时间">{{ formatTime(detailAsset.created_at) }}</el-descriptions-item>
              <el-descriptions-item label="更新时间">{{ formatTime(detailAsset.updated_at) }}</el-descriptions-item>
            </el-descriptions>
          </el-tab-pane>

          <el-tab-pane label="元数据" name="metadata">
            <div class="tab-toolbar">
              <el-input v-model="metaForm.key" placeholder="key" style="width: 160px" />
              <el-input v-model="metaForm.value" placeholder="value" style="width: 200px" />
              <el-input v-model="metaForm.source" placeholder="source" style="width: 130px" />
              <el-input-number
                v-model="metaForm.confidence"
                :min="0"
                :max="1"
                :step="0.1"
                :precision="2"
                style="width: 130px"
              />
              <el-button type="primary" :loading="metaAdding" @click="submitMetadata">添加</el-button>
            </div>
            <el-table :data="metadata" border size="small" max-height="460">
              <el-table-column prop="key" label="Key" min-width="120" show-overflow-tooltip />
              <el-table-column prop="value" label="Value" min-width="180" show-overflow-tooltip />
              <el-table-column prop="source" label="来源" width="100" />
              <el-table-column prop="confidence" label="置信度" width="90" />
              <el-table-column label="操作" width="80">
                <template #default="{ row }">
                  <el-button link type="danger" size="small" @click="removeMetadata(row)">删除</el-button>
                </template>
              </el-table-column>
            </el-table>
          </el-tab-pane>

          <el-tab-pane label="血缘" name="lineage">
            <div class="tab-toolbar">
              <el-input v-model="lineageForm.target_id" placeholder="target_id" style="width: 170px" />
              <el-select v-model="lineageForm.relation_type" style="width: 150px">
                <el-option label="derived_from" value="derived_from" />
                <el-option label="produced_by" value="produced_by" />
                <el-option label="depends_on" value="depends_on" />
              </el-select>
              <el-input v-model="lineageForm.transformation" placeholder="transformation" style="width: 180px" />
              <el-button type="primary" :loading="lineageAdding" @click="submitLineage">添加血缘</el-button>
            </div>
            <el-alert
              v-if="upstream.length"
              type="info"
              :closable="false"
              class="upstream-alert"
            >
              <template #title>
                上游资产：{{ upstream.join('、') }}
              </template>
            </el-alert>
            <el-table :data="lineage" border size="small" max-height="400">
              <el-table-column prop="source_id" label="源资产" min-width="150" show-overflow-tooltip />
              <el-table-column prop="relation_type" label="关系" width="120" />
              <el-table-column prop="target_id" label="目标资产" min-width="150" show-overflow-tooltip />
              <el-table-column prop="resource_type" label="资源类型" width="100" />
              <el-table-column prop="transformation" label="变换" min-width="140" show-overflow-tooltip />
            </el-table>
          </el-tab-pane>

          <el-tab-pane label="质量报告" name="quality">
            <div class="tab-toolbar quality-toolbar">
              <el-input
                v-model="qualityForm.data"
                type="textarea"
                :rows="2"
                placeholder='二维数组 JSON，如 [[1.0, 0.2], [0.9, 0.3]]'
                class="quality-data"
              />
              <div class="quality-options">
                <el-switch v-model="qualityForm.mass_field" active-text="质量守恒检查" />
                <el-input-number
                  v-model="qualityForm.mass_tol"
                  :min="0.000001"
                  :step="0.000001"
                  :precision="6"
                  :controls="false"
                  style="width: 150px"
                />
                <el-button type="primary" :loading="qualityRunning" @click="submitQualityCheck">
                  运行检查
                </el-button>
              </div>
            </div>
            <el-empty v-if="!reports.length" description="暂无质量报告" :image-size="80" />
            <div v-for="(report, idx) in reports" :key="idx" class="report-card">
              <div class="report-header">
                <span class="report-time">{{ formatTime(report.created_at) }}</span>
                <el-tag :type="reportStatusType(report.status)" size="small">{{ report.status }}</el-tag>
                <el-progress
                  class="report-score"
                  :percentage="report.overall_score"
                  :color="qualityColor(report.overall_score)"
                  :stroke-width="12"
                  :text-inside="true"
                />
              </div>
              <div class="report-checks">
                <div
                  v-for="check in report.checks"
                  :key="check.check_name"
                  class="check-item"
                >
                  <el-icon :class="check.passed ? 'check-pass' : 'check-fail'">
                    <CircleCheck v-if="check.passed" />
                    <CircleClose v-else />
                  </el-icon>
                  <span class="check-name">{{ check.check_name }}</span>
                  <span class="check-detail">{{ check.detail }}</span>
                </div>
              </div>
            </div>
          </el-tab-pane>
        </el-tabs>
      </div>
    </el-drawer>
  </el-card>
</template>

<script setup lang="ts">
import { onMounted, reactive, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { CircleCheck, CircleClose, Plus, Refresh, Search } from '@element-plus/icons-vue'
import dayjs from 'dayjs'
import {
  addLineage,
  addMetadata,
  archiveAsset,
  deleteMetadata,
  getLineage,
  listAssets,
  listMetadata,
  listQualityReports,
  registerAsset,
  runQualityCheck,
  type Asset,
  type AssetListQuery,
  type LineageEdge,
  type MetadataEntry,
  type QualityReport,
} from '@/api/data'

// ---------------------------------------------------------------------------
// 列表 + 过滤 + 分页
// ---------------------------------------------------------------------------

const loading = ref(false)
const assets = ref<Asset[]>([])
const total = ref(0)

const kindOptions = [
  { label: '全部类型', value: '' },
  { label: 'field_product', value: 'field_product' },
  { label: 'dataset', value: 'dataset' },
  { label: 'run', value: 'run' },
  { label: 'model', value: 'model' },
]

const statusOptions = [
  { label: 'active', value: 'active' },
  { label: 'archived', value: 'archived' },
]

const filters = reactive({
  kind: '',
  field_name: '',
  status: 'active',
})

const pagination = reactive({
  page: 1,
  pageSize: 20,
})

async function fetchAssets() {
  loading.value = true
  try {
    const params: AssetListQuery = {
      status: filters.status || 'active',
      limit: pagination.pageSize,
      offset: (pagination.page - 1) * pagination.pageSize,
    }
    if (filters.kind) params.kind = filters.kind
    if (filters.field_name) params.field_name = filters.field_name
    const res = await listAssets(params)
    assets.value = res.assets
    total.value = res.total
  } finally {
    loading.value = false
  }
}

function handleSearch() {
  pagination.page = 1
  fetchAssets()
}

function handleReset() {
  filters.kind = ''
  filters.field_name = ''
  filters.status = 'active'
  pagination.page = 1
  fetchAssets()
}

function handleSizeChange() {
  pagination.page = 1
  fetchAssets()
}

function formatTime(ts: number) {
  return dayjs(ts * 1000).format('YYYY-MM-DD HH:mm:ss')
}

function kindTagType(kind: string): 'primary' | 'success' | 'warning' | 'info' {
  switch (kind) {
    case 'field_product':
      return 'primary'
    case 'dataset':
      return 'success'
    case 'run':
      return 'warning'
    default:
      return 'info'
  }
}

function qualityColor(score: number): string {
  if (score >= 80) return '#67c23a'
  if (score >= 60) return '#e6a23c'
  return '#f56c6c'
}

// ---------------------------------------------------------------------------
// 登记资产
// ---------------------------------------------------------------------------

const registerVisible = ref(false)
const registerLoading = ref(false)

const registerForm = reactive({
  asset_id: '',
  name: '',
  kind: 'field_product',
  description: '',
  field_name: '',
  units: '',
  shape: '',
  dtype: '',
  tags: '',
  quality_score: 0,
  sensitivity_level: 'internal',
  source_run_id: '',
  status: 'active',
  version: '1.0.0',
})

function openRegister() {
  Object.assign(registerForm, {
    asset_id: '',
    name: '',
    kind: 'field_product',
    description: '',
    field_name: '',
    units: '',
    shape: '',
    dtype: '',
    tags: '',
    quality_score: 0,
    sensitivity_level: 'internal',
    source_run_id: '',
    status: 'active',
    version: '1.0.0',
  })
  registerVisible.value = true
}

async function submitRegister() {
  if (!registerForm.asset_id.trim() || !registerForm.name.trim()) {
    ElMessage.warning('请填写资产 ID 与名称')
    return
  }
  registerLoading.value = true
  try {
    await registerAsset({
      asset_id: registerForm.asset_id,
      name: registerForm.name,
      kind: registerForm.kind,
      description: registerForm.description,
      field_name: registerForm.field_name || null,
      units: registerForm.units || null,
      shape: registerForm.shape || null,
      dtype: registerForm.dtype || null,
      tags: registerForm.tags
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean),
      quality_score: registerForm.quality_score,
      sensitivity_level: registerForm.sensitivity_level,
      source_run_id: registerForm.source_run_id || null,
      status: registerForm.status,
      version: registerForm.version,
    })
    ElMessage.success('资产已登记')
    registerVisible.value = false
    await fetchAssets()
  } finally {
    registerLoading.value = false
  }
}

async function handleArchive(asset: Asset) {
  try {
    await ElMessageBox.confirm(`确定归档资产「${asset.name}」吗？`, '归档确认', {
      type: 'warning',
      confirmButtonText: '归档',
      cancelButtonText: '取消',
    })
  } catch {
    return
  }
  await archiveAsset(asset.asset_id)
  ElMessage.success('已归档')
  await fetchAssets()
}

// ---------------------------------------------------------------------------
// 详情（元数据 / 血缘 / 质量报告）
// ---------------------------------------------------------------------------

const drawerVisible = ref(false)
const detailLoading = ref(false)
const detailAsset = ref<Asset | null>(null)
const activeTab = ref('info')
const drawerTitle = ref('资产详情')

const metadata = ref<MetadataEntry[]>([])
const lineage = ref<LineageEdge[]>([])
const upstream = ref<string[]>([])
const reports = ref<QualityReport[]>([])

async function openDetail(asset: Asset) {
  detailAsset.value = asset
  drawerTitle.value = `资产详情 — ${asset.name}`
  activeTab.value = 'info'
  drawerVisible.value = true
  await loadDetail(asset.asset_id)
}

async function loadDetail(assetId: string) {
  detailLoading.value = true
  try {
    const [meta, lin, rep] = await Promise.all([
      listMetadata(assetId),
      getLineage(assetId),
      listQualityReports(assetId),
    ])
    metadata.value = meta
    lineage.value = lin.lineage
    upstream.value = lin.upstream
    reports.value = rep
  } finally {
    detailLoading.value = false
  }
}

// 元数据
const metaForm = reactive({ key: '', value: '', source: 'manual', confidence: 1 })
const metaAdding = ref(false)

async function submitMetadata() {
  if (!detailAsset.value) return
  if (!metaForm.key.trim()) {
    ElMessage.warning('请输入 key')
    return
  }
  metaAdding.value = true
  try {
    await addMetadata(detailAsset.value.asset_id, {
      key: metaForm.key,
      value: metaForm.value,
      source: metaForm.source || 'manual',
      confidence: metaForm.confidence,
    })
    ElMessage.success('元数据已添加')
    metaForm.key = ''
    metaForm.value = ''
    metadata.value = await listMetadata(detailAsset.value.asset_id)
  } finally {
    metaAdding.value = false
  }
}

async function removeMetadata(entry: MetadataEntry) {
  if (!detailAsset.value) return
  await deleteMetadata(detailAsset.value.asset_id, entry.key)
  ElMessage.success('元数据已删除')
  metadata.value = await listMetadata(detailAsset.value.asset_id)
}

// 血缘
const lineageForm = reactive({
  target_id: '',
  relation_type: 'derived_from',
  transformation: '',
  resource_type: 'product',
})
const lineageAdding = ref(false)

async function submitLineage() {
  if (!detailAsset.value) return
  if (!lineageForm.target_id.trim()) {
    ElMessage.warning('请输入 target_id')
    return
  }
  lineageAdding.value = true
  try {
    await addLineage(detailAsset.value.asset_id, {
      target_id: lineageForm.target_id,
      relation_type: lineageForm.relation_type,
      transformation: lineageForm.transformation,
      resource_type: lineageForm.resource_type,
    })
    ElMessage.success('血缘已添加')
    lineageForm.target_id = ''
    lineageForm.transformation = ''
    const lin = await getLineage(detailAsset.value.asset_id)
    lineage.value = lin.lineage
    upstream.value = lin.upstream
  } finally {
    lineageAdding.value = false
  }
}

// 质量
const qualityForm = reactive({ data: '', mass_field: false, mass_tol: 1e-6 })
const qualityRunning = ref(false)

function reportStatusType(status: string): 'success' | 'warning' | 'danger' | 'info' {
  switch (status) {
    case 'passed':
      return 'success'
    case 'warning':
      return 'warning'
    case 'failed':
      return 'danger'
    default:
      return 'info'
  }
}

async function submitQualityCheck() {
  if (!detailAsset.value) return
  let data: unknown
  try {
    data = JSON.parse(qualityForm.data)
  } catch {
    ElMessage.warning('数据格式错误，请输入合法的 JSON 二维数组')
    return
  }
  if (!Array.isArray(data)) {
    ElMessage.warning('数据必须是二维数组')
    return
  }
  qualityRunning.value = true
  try {
    const res = await runQualityCheck({
      asset_id: detailAsset.value.asset_id,
      data: data as number[][],
      mass_field: qualityForm.mass_field,
      mass_tol: qualityForm.mass_tol,
    })
    ElMessage.success(`质量检查完成，得分 ${res.overall_score}`)
    reports.value = await listQualityReports(detailAsset.value.asset_id)
    await fetchAssets()
  } finally {
    qualityRunning.value = false
  }
}

onMounted(fetchAssets)
</script>

<style scoped>
.toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  flex-wrap: wrap;
  gap: 8px;
}
.filters {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  align-items: center;
}
.actions {
  display: flex;
  gap: 8px;
}
.pagination {
  display: flex;
  justify-content: flex-end;
  margin-top: 16px;
}
.detail-body {
  min-height: 200px;
}
.tab-toolbar {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  align-items: center;
  margin-bottom: 12px;
}
.tag-gap {
  margin-right: 6px;
}
.upstream-alert {
  margin-bottom: 12px;
}
.quality-toolbar {
  align-items: flex-start;
}
.quality-data {
  flex: 1;
  min-width: 240px;
}
.quality-options {
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
}
.report-card {
  border: 1px solid #ebeef5;
  border-radius: 6px;
  padding: 12px;
  margin-bottom: 12px;
}
.report-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 8px;
}
.report-time {
  color: #909399;
  font-size: 12px;
}
.report-score {
  flex: 1;
}
.report-checks {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.check-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
}
.check-pass {
  color: #67c23a;
}
.check-fail {
  color: #f56c6c;
}
.check-name {
  font-weight: 600;
  min-width: 130px;
}
.check-detail {
  color: #606266;
}
</style>
