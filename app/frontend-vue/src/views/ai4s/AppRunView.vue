<template>
  <div class="app-run">
    <el-card>
      <template #header>
        <span>{{ t('ai4s.run.title') }}</span>
      </template>

      <el-form label-width="130px" label-position="left">
        <el-form-item :label="t('ai4s.run.app')" required>
          <el-select
            v-model="selectedApp"
            filterable
            :placeholder="t('ai4s.run.selectAppPlaceholder')"
            style="width: 100%"
            :loading="loadingApps"
          >
            <el-option
              v-for="app in apps"
              :key="app.name"
              :label="t('ai4s.run.appOptionLabel', { name: app.name, family: app.family, version: app.version })"
              :value="app.name"
            />
          </el-select>
        </el-form-item>

        <el-form-item :label="t('ai4s.run.runMode')">
          <el-radio-group v-model="runMode">
            <el-radio-button value="local">{{ t('ai4s.run.modes.local') }}</el-radio-button>
            <el-radio-button value="hpc">{{ t('ai4s.run.modes.hpc') }}</el-radio-button>
          </el-radio-group>
        </el-form-item>

        <el-form-item v-if="runMode === 'hpc'" :label="t('ai4s.run.hpcParams')">
          <div class="hpc-grid">
            <div class="hpc-field">
              <label>partition</label>
              <el-input v-model="hpcForm.partition" />
            </div>
            <div class="hpc-field">
              <label>nodes</label>
              <el-input-number v-model="hpcForm.nodes" :min="1" style="width: 100%" />
            </div>
            <div class="hpc-field">
              <label>cpus</label>
              <el-input-number v-model="hpcForm.cpus" :min="1" style="width: 100%" />
            </div>
            <div class="hpc-field">
              <label>mem</label>
              <el-input v-model="hpcForm.mem" />
            </div>
            <div class="hpc-field">
              <label>walltime</label>
              <el-input v-model="hpcForm.walltime" />
            </div>
            <div class="hpc-field">
              <label>backend</label>
              <el-input v-model="hpcForm.backend" />
            </div>
          </div>
        </el-form-item>

        <el-form-item label="produce_cfg">
          <el-input
            v-model="produceCfgText"
            type="textarea"
            :rows="5"
            :placeholder="t('ai4s.run.produceCfgPlaceholder')"
          />
        </el-form-item>

        <el-form-item label="train_cfg">
          <el-input
            v-model="trainCfgText"
            type="textarea"
            :rows="5"
            :placeholder="t('ai4s.run.trainCfgPlaceholder')"
          />
        </el-form-item>

        <el-form-item>
          <el-button type="primary" :loading="submitting" @click="submit">{{ t('ai4s.run.submit') }}</el-button>
          <el-button @click="reset">{{ t('ai4s.run.reset') }}</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- 本地全栈 RunReport 结果 -->
    <el-card v-if="report" class="result-card">
      <template #header>
        <div class="card-header">
          <span>{{ t('ai4s.run.resultTitle') }}</span>
          <el-button text type="primary" @click="goLineage">{{ t('ai4s.run.viewLineage') }}</el-button>
        </div>
      </template>

      <el-descriptions :column="2" border>
        <el-descriptions-item :label="t('ai4s.run.app')">{{ report.name }}</el-descriptions-item>
        <el-descriptions-item :label="t('ai4s.run.family')">{{ report.family }}</el-descriptions-item>
        <el-descriptions-item label="data_asset_id">{{ report.data_asset_id }}</el-descriptions-item>
        <el-descriptions-item label="dataset_asset_id">{{ report.dataset_asset_id }}</el-descriptions-item>
        <el-descriptions-item label="job_id">{{ report.job_id }}</el-descriptions-item>
        <el-descriptions-item label="model_id">{{ report.model_id }}</el-descriptions-item>
      </el-descriptions>

      <h4 class="section-title">{{ t('ai4s.run.metricsTitle') }}</h4>
      <el-table :data="metricEntries" border size="small">
        <el-table-column prop="key" :label="t('ai4s.run.metric')" min-width="180" />
        <el-table-column :label="t('ai4s.run.value')" min-width="220">
          <template #default="{ row }">{{ formatValue(row.value) }}</template>
        </el-table-column>
        <template #empty>
          <el-empty :description="t('ai4s.run.noMetrics')" :image-size="60" />
        </template>
      </el-table>

      <h4 class="section-title">{{ t('ai4s.run.upstreamTitle') }}</h4>
      <div class="tag-row">
        <el-tag v-for="item in report.lineage_upstream" :key="item" type="info" class="lineage-tag">
          {{ item }}
        </el-tag>
        <el-text v-if="!report.lineage_upstream.length" type="info">{{ t('ai4s.run.noUpstream') }}</el-text>
      </div>
    </el-card>

    <!-- HPC 派发结果 -->
    <el-card v-else-if="hpcResult" class="result-card">
      <template #header>
        <span>{{ t('ai4s.run.hpcResultTitle') }}</span>
      </template>

      <el-descriptions :column="2" border>
        <el-descriptions-item label="app_name">{{ hpcResult.app_name }}</el-descriptions-item>
        <el-descriptions-item label="status">{{ hpcResult.status }}</el-descriptions-item>
        <el-descriptions-item label="job_id">{{ hpcResult.job_id }}</el-descriptions-item>
        <el-descriptions-item label="hpc_job_id">{{ hpcResult.hpc_job_id }}</el-descriptions-item>
        <el-descriptions-item label="backend">{{ hpcResult.backend }}</el-descriptions-item>
        <el-descriptions-item label="script_cmd">{{ hpcResult.script_cmd }}</el-descriptions-item>
      </el-descriptions>

      <div class="status-actions">
        <el-button type="primary" :loading="queryingStatus" @click="queryStatus">{{ t('ai4s.run.queryStatus') }}</el-button>
      </div>

      <el-descriptions v-if="statusResult" :column="2" border class="status-result">
        <el-descriptions-item label="job_id">{{ statusResult.job_id }}</el-descriptions-item>
        <el-descriptions-item label="app_name">{{ statusResult.app_name }}</el-descriptions-item>
        <el-descriptions-item label="status">{{ statusResult.status }}</el-descriptions-item>
        <el-descriptions-item label="scheduler_state">{{ statusResult.scheduler_state ?? '-' }}</el-descriptions-item>
        <el-descriptions-item label="hpc_job_id">{{ statusResult.hpc_job_id ?? '-' }}</el-descriptions-item>
        <el-descriptions-item label="elapsed">{{ statusResult.elapsed ?? '-' }}</el-descriptions-item>
      </el-descriptions>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'
import {
  getRunStatus,
  isHpcSubmitResponse,
  isRunReport,
  listApps,
  runApp,
} from '@/api/apps'
import type {
  AppInfo,
  AppRunRequest,
  HpcRequest,
  HpcSubmitResponse,
  RunReportOut,
  RunStatusResponse,
} from '@/api/apps'
import { useRunHistory } from '@/stores/ai4sRun'

const route = useRoute()
const router = useRouter()
const { t } = useI18n()
const { save } = useRunHistory()

const loadingApps = ref(false)
const apps = ref<AppInfo[]>([])
const selectedApp = ref('')
const runMode = ref<'local' | 'hpc'>('local')
const produceCfgText = ref('{}')
const trainCfgText = ref('{}')
const submitting = ref(false)

const hpcForm = reactive<Required<HpcRequest>>({
  partition: 'compute',
  nodes: 1,
  cpus: 4,
  mem: '8G',
  walltime: '02:00:00',
  backend: 'slurm',
})

const report = ref<RunReportOut | null>(null)
const hpcResult = ref<HpcSubmitResponse | null>(null)
const statusResult = ref<RunStatusResponse | null>(null)
const queryingStatus = ref(false)

const metricEntries = computed(() =>
  report.value ? Object.entries(report.value.metrics ?? {}).map(([key, value]) => ({ key, value })) : [],
)

async function fetchApps() {
  loadingApps.value = true
  try {
    const res = await listApps()
    apps.value = res.apps ?? []
    preselectFromRoute()
  } catch {
    // 错误提示已由请求拦截器统一处理
  } finally {
    loadingApps.value = false
  }
}

function preselectFromRoute() {
  const wanted = Array.isArray(route.params.name) ? route.params.name[0] : route.params.name
  if (wanted && apps.value.some((app) => app.name === wanted)) {
    selectedApp.value = wanted
  }
}

watch(
  () => route.params.name,
  () => preselectFromRoute(),
)

function parseJsonObject(text: string, field: string): Record<string, unknown> | null {
  const trimmed = text.trim()
  if (!trimmed) return {}
  try {
    const parsed = JSON.parse(trimmed)
    if (parsed === null || Array.isArray(parsed) || typeof parsed !== 'object') {
      ElMessage.error(t('ai4s.run.messages.mustBeJsonObject', { field }))
      return null
    }
    return parsed as Record<string, unknown>
  } catch {
    ElMessage.error(t('ai4s.run.messages.invalidJson', { field }))
    return null
  }
}

async function submit() {
  if (!selectedApp.value) {
    ElMessage.warning(t('ai4s.run.messages.selectAppFirst'))
    return
  }
  const produceCfg = parseJsonObject(produceCfgText.value, 'produce_cfg')
  const trainCfg = parseJsonObject(trainCfgText.value, 'train_cfg')
  if (produceCfg === null || trainCfg === null) return

  report.value = null
  hpcResult.value = null
  statusResult.value = null

  const body: AppRunRequest = {
    produce_cfg: produceCfg,
    train_cfg: trainCfg,
    hpc: runMode.value === 'hpc' ? { ...hpcForm } : null,
  }

  submitting.value = true
  try {
    const res = await runApp(selectedApp.value, body)
    if (isRunReport(res)) {
      report.value = res
      save(res)
      ElMessage.success(t('ai4s.run.messages.runDone'))
    } else if (isHpcSubmitResponse(res)) {
      hpcResult.value = res
      ElMessage.success(t('ai4s.run.messages.hpcSubmitted'))
    }
  } catch {
    // 错误提示已由请求拦截器统一处理
  } finally {
    submitting.value = false
  }
}

async function queryStatus() {
  if (!hpcResult.value) return
  queryingStatus.value = true
  try {
    statusResult.value = await getRunStatus(hpcResult.value.app_name, hpcResult.value.job_id)
  } catch {
    // 错误提示已由请求拦截器统一处理
  } finally {
    queryingStatus.value = false
  }
}

function reset() {
  produceCfgText.value = '{}'
  trainCfgText.value = '{}'
  report.value = null
  hpcResult.value = null
  statusResult.value = null
}

function goLineage() {
  router.push('/ai4s/lineage')
}

function formatValue(value: unknown): string {
  if (typeof value === 'string') return value
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

onMounted(fetchApps)
</script>

<style scoped>
.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.result-card {
  margin-top: 16px;
}
.section-title {
  margin: 16px 0 8px;
}
.tag-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.lineage-tag {
  font-family: monospace;
}
.hpc-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  width: 100%;
}
.hpc-field label {
  display: block;
  margin-bottom: 4px;
  font-size: 12px;
  color: #909399;
}
.status-actions {
  margin-top: 16px;
}
.status-result {
  margin-top: 16px;
}
</style>
