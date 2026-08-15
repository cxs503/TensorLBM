<template>
  <el-card shadow="never">
    <!-- 过滤栏 -->
    <div class="toolbar">
      <div class="filters">
        <el-select
          v-model="filters.kind"
          :placeholder="t('data.filter.kindPlaceholder')"
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
          :placeholder="t('data.filter.fieldPlaceholder')"
          clearable
          style="width: 200px"
          @keyup.enter="handleSearch"
          @clear="handleSearch"
        />
        <el-select
          v-model="filters.status"
          :placeholder="t('data.filter.statusPlaceholder')"
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
        <el-button type="primary" :icon="Search" @click="handleSearch">{{ t('data.search') }}</el-button>
        <el-button :icon="Refresh" @click="handleReset">{{ t('data.reset') }}</el-button>
      </div>
      <div class="actions">
        <el-button type="success" :icon="Plus" @click="openRegister">{{ t('data.registerAsset') }}</el-button>
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
      <el-table-column prop="asset_id" :label="t('data.columns.assetId')" min-width="180" show-overflow-tooltip />
      <el-table-column prop="name" :label="t('data.columns.name')" min-width="140" show-overflow-tooltip />
      <el-table-column prop="kind" :label="t('data.columns.kind')" width="140">
        <template #default="{ row }">
          <el-tag :type="kindTagType(row.kind)" size="small">{{ row.kind }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="field_name" :label="t('data.columns.fieldName')" min-width="120" show-overflow-tooltip>
        <template #default="{ row }">
          <span>{{ row.field_name || '—' }}</span>
        </template>
      </el-table-column>
      <el-table-column prop="quality_score" :label="t('data.columns.qualityScore')" width="160">
        <template #default="{ row }">
          <el-progress
            :percentage="row.quality_score"
            :color="qualityColor(row.quality_score)"
            :stroke-width="14"
            :text-inside="true"
          />
        </template>
      </el-table-column>
      <el-table-column prop="status" :label="t('data.columns.status')" width="110">
        <template #default="{ row }">
          <el-tag :type="row.status === 'active' ? 'success' : 'info'" size="small">
            {{ row.status }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column :label="t('data.columns.updatedAt')" width="170">
        <template #default="{ row }">
          <span>{{ formatTime(row.updated_at) }}</span>
        </template>
      </el-table-column>
      <el-table-column :label="t('data.columns.actions')" width="150" fixed="right">
        <template #default="{ row }">
          <el-button link type="primary" size="small" @click.stop="openDetail(row)">{{ t('data.detail') }}</el-button>
          <el-button link type="danger" size="small" @click.stop="handleArchive(row)">{{ t('data.archive') }}</el-button>
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
      :title="t('data.registerAsset')"
      width="640px"
      :close-on-click-modal="false"
    >
      <el-form :model="registerForm" label-width="120px">
        <el-row :gutter="16">
          <el-col :span="12">
            <el-form-item :label="t('data.form.assetId')" required>
              <el-input v-model="registerForm.asset_id" :placeholder="t('data.form.assetIdPlaceholder')" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('data.form.name')" required>
              <el-input v-model="registerForm.name" :placeholder="t('data.form.assetNamePlaceholder')" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('data.form.kind')">
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
            <el-form-item :label="t('data.form.fieldName')">
              <el-input v-model="registerForm.field_name" placeholder="field_name" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('data.form.units')">
              <el-input v-model="registerForm.units" placeholder="units" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('data.form.dtype')">
              <el-input v-model="registerForm.dtype" placeholder="float32" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('data.form.shape')">
              <el-input v-model="registerForm.shape" placeholder="[128, 128]" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('data.form.version')">
              <el-input v-model="registerForm.version" placeholder="1.0.0" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('data.form.sourceRunId')">
              <el-input v-model="registerForm.source_run_id" placeholder="source_run_id" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('data.form.sensitivityLevel')">
              <el-select v-model="registerForm.sensitivity_level" style="width: 100%">
                <el-option label="internal" value="internal" />
                <el-option label="public" value="public" />
                <el-option label="restricted" value="restricted" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('data.form.qualityScore')">
              <el-input-number
                v-model="registerForm.quality_score"
                :min="0"
                :max="100"
                style="width: 100%"
              />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('data.form.tags')">
              <el-input v-model="registerForm.tags" :placeholder="t('data.form.tagsPlaceholder')" />
            </el-form-item>
          </el-col>
          <el-col :span="24">
            <el-form-item :label="t('data.form.description')">
              <el-input v-model="registerForm.description" type="textarea" :rows="2" />
            </el-form-item>
          </el-col>
        </el-row>
      </el-form>
      <template #footer>
        <el-button @click="registerVisible = false">{{ t('data.cancel') }}</el-button>
        <el-button type="primary" :loading="registerLoading" @click="submitRegister">{{ t('data.submit') }}</el-button>
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
          <el-tab-pane :label="t('data.tabs.info')" name="info">
            <el-descriptions v-if="detailAsset" :column="1" border>
              <el-descriptions-item :label="t('data.form.assetId')">{{ detailAsset.asset_id }}</el-descriptions-item>
              <el-descriptions-item :label="t('data.form.name')">{{ detailAsset.name }}</el-descriptions-item>
              <el-descriptions-item :label="t('data.form.kind')">{{ detailAsset.kind }}</el-descriptions-item>
              <el-descriptions-item :label="t('data.form.fieldName')">{{ detailAsset.field_name || '—' }}</el-descriptions-item>
              <el-descriptions-item :label="t('data.form.units')">{{ detailAsset.units || '—' }}</el-descriptions-item>
              <el-descriptions-item :label="t('data.form.shape')">{{ detailAsset.shape || '—' }}</el-descriptions-item>
              <el-descriptions-item :label="t('data.form.dtype')">{{ detailAsset.dtype || '—' }}</el-descriptions-item>
              <el-descriptions-item :label="t('data.form.qualityScore')">{{ detailAsset.quality_score }}</el-descriptions-item>
              <el-descriptions-item :label="t('data.form.sensitivityLevel')">{{ detailAsset.sensitivity_level }}</el-descriptions-item>
              <el-descriptions-item :label="t('data.form.sourceRunId')">{{ detailAsset.source_run_id || '—' }}</el-descriptions-item>
              <el-descriptions-item :label="t('data.form.status')">{{ detailAsset.status }}</el-descriptions-item>
              <el-descriptions-item :label="t('data.form.version')">{{ detailAsset.version }}</el-descriptions-item>
              <el-descriptions-item :label="t('data.form.tags')">
                <el-tag v-for="t in detailAsset.tags" :key="t" size="small" class="tag-gap">
                  {{ t }}
                </el-tag>
                <span v-if="!detailAsset.tags.length">—</span>
              </el-descriptions-item>
              <el-descriptions-item :label="t('data.form.description')">{{ detailAsset.description || '—' }}</el-descriptions-item>
              <el-descriptions-item :label="t('data.form.createdTime')">{{ formatTime(detailAsset.created_at) }}</el-descriptions-item>
              <el-descriptions-item :label="t('data.form.updatedTime')">{{ formatTime(detailAsset.updated_at) }}</el-descriptions-item>
            </el-descriptions>
          </el-tab-pane>

          <el-tab-pane :label="t('data.tabs.metadata')" name="metadata">
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
              <el-button type="primary" :loading="metaAdding" @click="submitMetadata">{{ t('data.metadata.add') }}</el-button>
            </div>
            <el-table :data="metadata" border size="small" max-height="460">
              <el-table-column prop="key" label="Key" min-width="120" show-overflow-tooltip />
              <el-table-column prop="value" label="Value" min-width="180" show-overflow-tooltip />
              <el-table-column prop="source" :label="t('data.metadata.source')" width="100" />
              <el-table-column prop="confidence" :label="t('data.metadata.confidence')" width="90" />
              <el-table-column :label="t('data.columns.actions')" width="80">
                <template #default="{ row }">
                  <el-button link type="danger" size="small" @click="removeMetadata(row)">{{ t('data.metadata.delete') }}</el-button>
                </template>
              </el-table-column>
            </el-table>
          </el-tab-pane>

          <el-tab-pane :label="t('data.tabs.lineage')" name="lineage">
            <div class="tab-toolbar">
              <el-input v-model="lineageForm.target_id" placeholder="target_id" style="width: 170px" />
              <el-select v-model="lineageForm.relation_type" style="width: 150px">
                <el-option label="derived_from" value="derived_from" />
                <el-option label="produced_by" value="produced_by" />
                <el-option label="depends_on" value="depends_on" />
              </el-select>
              <el-input v-model="lineageForm.transformation" placeholder="transformation" style="width: 180px" />
              <el-button type="primary" :loading="lineageAdding" @click="submitLineage">{{ t('data.lineage.add') }}</el-button>
            </div>
            <el-alert
              v-if="upstream.length"
              type="info"
              :closable="false"
              class="upstream-alert"
            >
              <template #title>
                {{ t('data.lineage.upstream') }}：{{ upstream.join(listSep) }}
              </template>
            </el-alert>
            <el-table :data="lineage" border size="small" max-height="400">
              <el-table-column prop="source_id" :label="t('data.lineage.source')" min-width="150" show-overflow-tooltip />
              <el-table-column prop="relation_type" :label="t('data.lineage.relation')" width="120" />
              <el-table-column prop="target_id" :label="t('data.lineage.target')" min-width="150" show-overflow-tooltip />
              <el-table-column prop="resource_type" :label="t('data.lineage.resourceType')" width="100" />
              <el-table-column prop="transformation" :label="t('data.lineage.transformation')" min-width="140" show-overflow-tooltip />
            </el-table>
          </el-tab-pane>

          <el-tab-pane :label="t('data.tabs.quality')" name="quality">
            <div class="tab-toolbar quality-toolbar">
              <el-input
                v-model="qualityForm.data"
                type="textarea"
                :rows="2"
                :placeholder="t('data.quality.dataPlaceholder')"
                class="quality-data"
              />
              <div class="quality-options">
                <el-switch v-model="qualityForm.mass_field" :active-text="t('data.quality.massCheck')" />
                <el-input-number
                  v-model="qualityForm.mass_tol"
                  :min="0.000001"
                  :step="0.000001"
                  :precision="6"
                  :controls="false"
                  style="width: 150px"
                />
                <el-button type="primary" :loading="qualityRunning" @click="submitQualityCheck">
                  {{ t('data.quality.run') }}
                </el-button>
              </div>
            </div>
            <el-empty v-if="!reports.length" :description="t('data.quality.empty')" :image-size="80" />
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
import { computed, onMounted, reactive, ref } from 'vue'
import { useI18n } from 'vue-i18n'
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

const { t, locale } = useI18n()

// ---------------------------------------------------------------------------
// 列表 + 过滤 + 分页
// ---------------------------------------------------------------------------

const loading = ref(false)
const assets = ref<Asset[]>([])
const total = ref(0)

const kindOptions = computed(() => [
  { label: t('data.kindAll'), value: '' },
  { label: 'field_product', value: 'field_product' },
  { label: 'dataset', value: 'dataset' },
  { label: 'run', value: 'run' },
  { label: 'model', value: 'model' },
])

const statusOptions = [
  { label: 'active', value: 'active' },
  { label: 'archived', value: 'archived' },
]

const listSep = computed(() => (locale.value === 'zh' ? '、' : ', '))

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
    ElMessage.warning(t('data.messages.fillAssetIdAndName'))
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
    ElMessage.success(t('data.messages.assetRegistered'))
    registerVisible.value = false
    await fetchAssets()
  } finally {
    registerLoading.value = false
  }
}

async function handleArchive(asset: Asset) {
  try {
    await ElMessageBox.confirm(t('data.messages.confirmArchive', { name: asset.name }), t('data.messages.confirmArchiveTitle'), {
      type: 'warning',
      confirmButtonText: t('data.archive'),
      cancelButtonText: t('data.cancel'),
    })
  } catch {
    return
  }
  await archiveAsset(asset.asset_id)
  ElMessage.success(t('data.messages.archived'))
  await fetchAssets()
}

// ---------------------------------------------------------------------------
// 详情（元数据 / 血缘 / 质量报告）
// ---------------------------------------------------------------------------

const drawerVisible = ref(false)
const detailLoading = ref(false)
const detailAsset = ref<Asset | null>(null)
const activeTab = ref('info')
const drawerTitle = computed(() =>
  detailAsset.value ? t('data.assetDetailOf', { name: detailAsset.value.name }) : t('data.assetDetail'),
)

const metadata = ref<MetadataEntry[]>([])
const lineage = ref<LineageEdge[]>([])
const upstream = ref<string[]>([])
const reports = ref<QualityReport[]>([])

async function openDetail(asset: Asset) {
  detailAsset.value = asset
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
    ElMessage.warning(t('data.messages.enterKey'))
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
    ElMessage.success(t('data.messages.metadataAdded'))
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
  ElMessage.success(t('data.messages.metadataDeleted'))
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
    ElMessage.warning(t('data.messages.enterTargetId'))
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
    ElMessage.success(t('data.messages.lineageAdded'))
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
    ElMessage.warning(t('data.messages.invalidData'))
    return
  }
  if (!Array.isArray(data)) {
    ElMessage.warning(t('data.messages.dataMustBe2d'))
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
    ElMessage.success(t('data.messages.qualityDone', { score: res.overall_score }))
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
