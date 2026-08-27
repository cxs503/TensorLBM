<template>
  <div>
    <el-row :gutter="16">
      <el-col :span="9">
        <el-card>
          <template #header><span>{{ t('production.solve.title') }}</span></template>
          <el-form label-width="130px" size="small">
            <el-form-item :label="t('production.solve.solverType')">
              <el-select v-model="selectedType" filterable style="width: 100%" @change="onTypeChange">
                <el-option
                  v-for="(s, key) in solverTypes"
                  :key="key"
                  :label="t(s.labelKey)"
                  :value="key"
                />
              </el-select>
            </el-form-item>
          </el-form>
          <el-alert
            v-if="currentDesc"
            :title="currentDesc"
            type="info"
            :closable="false"
            class="desc-alert"
          />

          <el-form label-width="150px" size="small">
            <el-form-item v-for="f in currentFields" :key="f.name" :label="t(f.labelKey)">
              <el-input-number
                v-if="f.type === 'number'"
                v-model="formData[f.name]"
                :min="f.min"
                :max="f.max"
                :step="f.step"
                :precision="f.precision"
                controls-position="right"
                style="width: 100%"
              />
              <el-select v-else-if="f.type === 'select'" v-model="formData[f.name]" style="width: 100%">
                <el-option v-for="o in f.options" :key="o" :label="o" :value="o" />
              </el-select>
              <el-input v-else v-model="formData[f.name]" />
            </el-form-item>
            <el-form-item>
              <el-button type="primary" :loading="submitting" @click="submitJob">
                {{ t('production.solve.submitJob') }}
              </el-button>
              <el-button @click="runPreflight" :loading="preflightLoading">{{ t('production.solve.preflight') }}</el-button>
            </el-form-item>
          </el-form>

          <el-alert
            v-if="preflightResult"
            :title="preflightTitle"
            :type="preflightType"
            :closable="true"
            class="desc-alert"
          >
            <ul class="preflight-list">
              <li v-for="(c, i) in preflightChecks" :key="i">
                <strong>{{ c.name.replace(/_/g, ' ') }}</strong>：{{ c.message }}
              </li>
            </ul>
          </el-alert>
        </el-card>
      </el-col>

      <el-col :span="15">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>{{ t('production.solve.jobMonitor') }}</span>
              <el-button size="small" :icon="Refresh" @click="loadJobs">{{ t('production.solve.refreshJobList') }}</el-button>
            </div>
          </template>

          <!-- 当前提交的作业状态 -->
          <el-alert
            v-if="activeJob"
            :title="t('production.solve.activeJobTitle', { id: activeJob.job_id, status: statusLabel(activeJob.status) })"
            :type="statusAlertType(activeJob.status)"
            :closable="false"
            class="active-job-alert"
          >
            <template #default>
              <div>{{ t('production.solve.nameLabel', { name: activeJob.name }) }}</div>
              <el-progress
                v-if="activeJob.status === 'running'"
                :percentage="progressPercent"
                :stroke-width="14"
                class="progress-bar"
              />
              <div v-if="activeJob.error" class="error-text">{{ activeJob.error }}</div>
            </template>
          </el-alert>

          <el-table :data="jobs" v-loading="jobsLoading" stripe size="small" style="width: 100%">
            <el-table-column prop="name" :label="t('production.common.name')" min-width="180" show-overflow-tooltip />
            <el-table-column prop="job_id" :label="t('production.common.id')" width="100" />
            <el-table-column :label="t('production.common.status')" width="100">
              <template #default="{ row }">
                <el-tag :type="statusTagType(row.status)" disable-transitions>{{ statusLabel(row.status) }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column :label="t('production.common.createdAt')" width="170">
              <template #default="{ row }">
                <span class="muted">{{ formatTime(row.created_at) }}</span>
              </template>
            </el-table-column>
            <el-table-column :label="t('production.common.action')" width="110">
              <template #default="{ row }">
                <el-button
                  v-if="['queued', 'running'].includes(row.status)"
                  size="small"
                  type="warning"
                  @click="cancel(row)"
                >
                  {{ t('production.common.cancel') }}
                </el-button>
                <el-button size="small" type="primary" link @click="goPostprocess(row)">
                  {{ t('production.common.postprocess') }}
                </el-button>
              </template>
            </el-table-column>
            <template #empty><el-empty :description="t('production.common.noJobs')" :image-size="60" /></template>
          </el-table>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Refresh } from '@element-plus/icons-vue'
import { useI18n } from 'vue-i18n'
import {
  cancelJob,
  getJob,
  getLiveMetrics,
  listJobs,
  preprocessPreflight,
  submitSolverJob,
  type Job,
} from '@/api/production'

const { t } = useI18n()

// ---------------------------------------------------------------------------
// 求解器类型配置（字段与后端 Pydantic schema 对应）
// ---------------------------------------------------------------------------

interface SolverField {
  name: string
  labelKey: string
  type: 'number' | 'select' | 'text'
  default?: any
  min?: number
  max?: number
  step?: number
  precision?: number
  options?: string[]
}

interface SolverTypeDef {
  labelKey: string
  descKey: string
  endpoint: string
  fields: SolverField[]
}

const solverTypes: Record<string, SolverTypeDef> = {
  cylinder_flow: {
    labelKey: 'production.solve.labelCylinderFlow',
    descKey: 'production.solve.descCylinderFlow',
    endpoint: '/solve/cylinder-flow',
    fields: [
      { name: 'nx', labelKey: 'production.fields.meshWideNx', type: 'number', default: 320, min: 20 },
      { name: 'ny', labelKey: 'production.fields.meshHighNy', type: 'number', default: 100, min: 10 },
      { name: 'u_in', labelKey: 'production.fields.uIn', type: 'number', default: 0.08, step: 0.01, min: 0.001 },
      { name: 're', labelKey: 'production.fields.re', type: 'number', default: 100, min: 1 },
      { name: 'radius', labelKey: 'production.fields.cylinderRadius', type: 'number', default: 12, min: 1 },
      { name: 'n_steps', labelKey: 'production.fields.nSteps', type: 'number', default: 1200, min: 1 },
      { name: 'output_interval', labelKey: 'production.fields.outputInterval', type: 'number', default: 200, min: 1 },
      { name: 'device', labelKey: 'production.fields.device', type: 'select', default: 'cpu', options: ['cpu', 'cuda:0'] },
      { name: 'seed', labelKey: 'production.fields.seed', type: 'number', default: 0, min: 0 },
    ],
  },
  rotating_cylinder: {
    labelKey: 'production.solve.labelRotatingCylinder',
    descKey: 'production.solve.descRotatingCylinder',
    endpoint: '/solve/rotating-cylinder',
    fields: [
      { name: 'nx', labelKey: 'production.fields.meshWideNx', type: 'number', default: 320, min: 16 },
      { name: 'ny', labelKey: 'production.fields.meshHighNy', type: 'number', default: 100, min: 8 },
      { name: 'u_in', labelKey: 'production.fields.uIn', type: 'number', default: 0.08, step: 0.01, min: 0.001 },
      { name: 're', labelKey: 'production.fields.re', type: 'number', default: 100, min: 1 },
      { name: 'radius', labelKey: 'production.fields.cylinderRadiusShort', type: 'number', default: 12, min: 1 },
      { name: 'spin_ratio', labelKey: 'production.fields.spinRatio', type: 'number', default: 1.0, step: 0.1, min: 0 },
      { name: 'n_steps', labelKey: 'production.fields.nSteps', type: 'number', default: 1200, min: 1 },
      { name: 'output_interval', labelKey: 'production.fields.outputInterval', type: 'number', default: 200, min: 1 },
      { name: 'device', labelKey: 'production.fields.device', type: 'select', default: 'cpu', options: ['cpu', 'cuda:0'] },
      { name: 'seed', labelKey: 'production.fields.seed', type: 'number', default: 0, min: 0 },
    ],
  },
  lid_driven_cavity: {
    labelKey: 'production.solve.labelLidDrivenCavity',
    descKey: 'production.solve.descLidDrivenCavity',
    endpoint: '/solve/lid-driven-cavity',
    fields: [
      { name: 'nx', labelKey: 'production.fields.meshSizeNx', type: 'number', default: 128, min: 8 },
      { name: 'u_lid', labelKey: 'production.fields.lidSpeed', type: 'number', default: 0.1, step: 0.01, min: 0.001 },
      { name: 're', labelKey: 'production.fields.re', type: 'number', default: 100, min: 1 },
      { name: 'n_steps', labelKey: 'production.fields.nSteps', type: 'number', default: 10000, min: 1 },
      { name: 'output_interval', labelKey: 'production.fields.outputInterval', type: 'number', default: 2000, min: 1 },
      { name: 'device', labelKey: 'production.fields.device', type: 'select', default: 'cpu', options: ['cpu', 'cuda:0'] },
      { name: 'seed', labelKey: 'production.fields.seed', type: 'number', default: 0, min: 0 },
    ],
  },
  backward_facing_step: {
    labelKey: 'production.solve.labelBackwardFacingStep',
    descKey: 'production.solve.descBackwardFacingStep',
    endpoint: '/solve/backward-facing-step',
    fields: [
      { name: 'nx', labelKey: 'production.fields.nx', type: 'number', default: 400, min: 20 },
      { name: 'ny', labelKey: 'production.fields.ny', type: 'number', default: 80, min: 6 },
      { name: 'step_h', labelKey: 'production.fields.stepHeight', type: 'number', default: 40, min: 1 },
      { name: 'x_step', labelKey: 'production.fields.stepLength', type: 'number', default: 80, min: 1 },
      { name: 'u_in', labelKey: 'production.fields.uIn', type: 'number', default: 0.05, step: 0.01 },
      { name: 're', labelKey: 'production.fields.re', type: 'number', default: 100, min: 1 },
      { name: 'n_steps', labelKey: 'production.fields.nSteps', type: 'number', default: 30000, min: 1 },
      { name: 'output_interval', labelKey: 'production.fields.outputInterval', type: 'number', default: 5000, min: 1 },
      { name: 'device', labelKey: 'production.fields.device', type: 'select', default: 'cpu', options: ['cpu', 'cuda:0'] },
    ],
  },
  turbulent_channel: {
    labelKey: 'production.solve.labelTurbulentChannel',
    descKey: 'production.solve.descTurbulentChannel',
    endpoint: '/solve/turbulent-channel',
    fields: [
      { name: 'nx', labelKey: 'production.fields.nx', type: 'number', default: 256, min: 16 },
      { name: 'ny', labelKey: 'production.fields.ny', type: 'number', default: 64, min: 8 },
      { name: 're_tau', labelKey: 'production.fields.reTau', type: 'number', default: 100, min: 1 },
      { name: 'u_tau', labelKey: 'production.fields.uTau', type: 'number', default: 0.005, step: 0.001, min: 0.0001 },
      { name: 'smagorinsky_cs', labelKey: 'production.fields.smagorinskyCs', type: 'number', default: 0.1, step: 0.01 },
      { name: 'n_steps', labelKey: 'production.fields.nSteps', type: 'number', default: 50000, min: 1 },
      { name: 'averaging_start', labelKey: 'production.fields.averagingStart', type: 'number', default: 20000, min: 0 },
      { name: 'output_interval', labelKey: 'production.fields.outputInterval', type: 'number', default: 5000, min: 1 },
      { name: 'device', labelKey: 'production.fields.device', type: 'select', default: 'cpu', options: ['cpu', 'cuda:0'] },
    ],
  },
  pipeline_flow: {
    labelKey: 'production.solve.labelPipelineFlow',
    descKey: 'production.solve.descPipelineFlow',
    endpoint: '/solve/pipeline-flow',
    fields: [
      { name: 'nx', labelKey: 'production.fields.nx', type: 'number', default: 400, min: 20 },
      { name: 'ny', labelKey: 'production.fields.ny', type: 'number', default: 160, min: 10 },
      { name: 'diameter', labelKey: 'production.fields.cylinderDiameter', type: 'number', default: 20, min: 2 },
      { name: 'gap_ratio', labelKey: 'production.fields.gapRatio', type: 'number', default: 0.5, step: 0.1, min: 0 },
      { name: 'u_in', labelKey: 'production.fields.uIn', type: 'number', default: 0.05, step: 0.01 },
      { name: 're', labelKey: 'production.fields.re', type: 'number', default: 200, min: 1 },
      { name: 'n_steps', labelKey: 'production.fields.nSteps', type: 'number', default: 30000, min: 1 },
      { name: 'output_interval', labelKey: 'production.fields.outputInterval', type: 'number', default: 5000, min: 1 },
      { name: 'device', labelKey: 'production.fields.device', type: 'select', default: 'cpu', options: ['cpu', 'cuda:0'] },
    ],
  },
  dam_break: {
    labelKey: 'production.solve.labelDamBreak',
    descKey: 'production.solve.descDamBreak',
    endpoint: '/solve/dam-break',
    fields: [
      { name: 'nx', labelKey: 'production.fields.nx', type: 'number', default: 400, min: 20 },
      { name: 'ny', labelKey: 'production.fields.ny', type: 'number', default: 200, min: 10 },
      { name: 'dam_width', labelKey: 'production.fields.damWidth', type: 'number', default: 100, min: 1 },
      { name: 'model', labelKey: 'production.fields.multiphaseModel', type: 'select', default: 'cg', options: ['sc', 'scmp', 'cg', 'fe'] },
      { name: 'rho_heavy', labelKey: 'production.fields.heavyDensity', type: 'number', default: 0.8, step: 0.1 },
      { name: 'rho_light', labelKey: 'production.fields.lightDensity', type: 'number', default: 0.4, step: 0.1 },
      { name: 'tau', labelKey: 'production.fields.tau', type: 'number', default: 1.0, step: 0.1, min: 0.51 },
      { name: 'n_steps', labelKey: 'production.fields.nSteps', type: 'number', default: 4000, min: 1 },
      { name: 'output_interval', labelKey: 'production.fields.outputInterval', type: 'number', default: 400, min: 1 },
      { name: 'device', labelKey: 'production.fields.device', type: 'select', default: 'cpu', options: ['cpu', 'cuda:0'] },
    ],
  },
  sloshing_tank: {
    labelKey: 'production.solve.labelSloshingTank',
    descKey: 'production.solve.descSloshingTank',
    endpoint: '/solve/sloshing-tank',
    fields: [
      { name: 'nx', labelKey: 'production.fields.nx', type: 'number', default: 200, min: 16 },
      { name: 'ny', labelKey: 'production.fields.ny', type: 'number', default: 160, min: 16 },
      { name: 'water_level', labelKey: 'production.fields.waterLevel', type: 'number', default: 80, min: 1 },
      { name: 'rho_water', labelKey: 'production.fields.waterDensity', type: 'number', default: 0.8, step: 0.1 },
      { name: 'rho_air', labelKey: 'production.fields.airDensity', type: 'number', default: 0.4, step: 0.1 },
      { name: 'tau', labelKey: 'production.fields.tau', type: 'number', default: 1.0, step: 0.1, min: 0.51 },
      { name: 'n_steps', labelKey: 'production.fields.nSteps', type: 'number', default: 6000, min: 1 },
      { name: 'output_interval', labelKey: 'production.fields.outputInterval', type: 'number', default: 600, min: 1 },
      { name: 'device', labelKey: 'production.fields.device', type: 'select', default: 'cpu', options: ['cpu', 'cuda:0'] },
    ],
  },
  sphere_flow: {
    labelKey: 'production.solve.labelSphereFlow',
    descKey: 'production.solve.descSphereFlow',
    endpoint: '/solve/sphere-flow',
    fields: [
      { name: 'nx', labelKey: 'production.fields.nx', type: 'number', default: 120, min: 20 },
      { name: 'ny', labelKey: 'production.fields.ny', type: 'number', default: 60, min: 10 },
      { name: 'nz', labelKey: 'production.fields.nz', type: 'number', default: 60, min: 10 },
      { name: 'u_in', labelKey: 'production.fields.uIn', type: 'number', default: 0.06, step: 0.01 },
      { name: 're', labelKey: 'production.fields.re', type: 'number', default: 50, min: 1 },
      { name: 'radius', labelKey: 'production.fields.sphereRadius', type: 'number', default: 8, min: 1 },
      { name: 'n_steps', labelKey: 'production.fields.nSteps', type: 'number', default: 500, min: 1 },
      { name: 'output_interval', labelKey: 'production.fields.outputInterval', type: 'number', default: 100, min: 1 },
      { name: 'device', labelKey: 'production.fields.device', type: 'select', default: 'cpu', options: ['cpu', 'cuda:0'] },
    ],
  },
  ship_hull: {
    labelKey: 'production.solve.labelShipHull',
    descKey: 'production.solve.descShipHull',
    endpoint: '/solve/ship-hull',
    fields: [
      { name: 'nx', labelKey: 'production.fields.nx', type: 'number', default: 160, min: 20 },
      { name: 'ny', labelKey: 'production.fields.ny', type: 'number', default: 60, min: 10 },
      { name: 'nz', labelKey: 'production.fields.nz', type: 'number', default: 40, min: 10 },
      { name: 'u_in', labelKey: 'production.fields.uIn', type: 'number', default: 0.05, step: 0.01 },
      { name: 're', labelKey: 'production.fields.re', type: 'number', default: 200, min: 1 },
      { name: 'hull_length', labelKey: 'production.fields.hullLength', type: 'number', default: 80, min: 10 },
      { name: 'hull_beam', labelKey: 'production.fields.hullBeam', type: 'number', default: 8, min: 1 },
      { name: 'hull_draft', labelKey: 'production.fields.hullDraft', type: 'number', default: 12, min: 1 },
      { name: 'smagorinsky_cs', labelKey: 'production.fields.smagorinskyCs', type: 'number', default: 0.1, step: 0.01 },
      { name: 'wave_amp', labelKey: 'production.fields.waveAmp', type: 'number', default: 0, step: 0.5, min: 0 },
      { name: 'n_steps', labelKey: 'production.fields.nSteps', type: 'number', default: 2000, min: 1 },
      { name: 'output_interval', labelKey: 'production.fields.outputInterval', type: 'number', default: 200, min: 1 },
      { name: 'device', labelKey: 'production.fields.device', type: 'select', default: 'cpu', options: ['cpu', 'cuda:0'] },
    ],
  },
  suboff_wall_function: {
    labelKey: 'production.solve.labelSuboff',
    descKey: 'production.solve.descSuboff',
    endpoint: '/solve/suboff-wall-function',
    fields: [
      { name: 're', labelKey: 'production.fields.re', type: 'number', default: 2000000, min: 1 },
      { name: 'hull_type', labelKey: 'production.fields.hullType', type: 'select', default: 'full', options: ['bare_hull', 'with_sail', 'full'] },
      { name: 'nx', labelKey: 'production.fields.nx', type: 'number', default: 320, min: 80 },
      { name: 'ny', labelKey: 'production.fields.ny', type: 'number', default: 128, min: 32 },
      { name: 'nz', labelKey: 'production.fields.nz', type: 'number', default: 128, min: 32 },
      { name: 'hull_length', labelKey: 'production.fields.hullLength', type: 'number', default: 128, min: 20 },
      { name: 'u_in', labelKey: 'production.fields.uIn', type: 'number', default: 0.06, step: 0.01 },
      { name: 'n_steps', labelKey: 'production.fields.nSteps', type: 'number', default: 5000, min: 100 },
      { name: 'output_interval', labelKey: 'production.fields.outputInterval', type: 'number', default: 500, min: 10 },
      { name: 'device', labelKey: 'production.fields.device', type: 'select', default: 'cpu', options: ['cpu', 'cuda:0'] },
    ],
  },
  porous_drainage: {
    labelKey: 'production.solve.labelPorousDrainage',
    descKey: 'production.solve.descPorousDrainage',
    endpoint: '/solve/porous-drainage',
    fields: [
      { name: 'nx', labelKey: 'production.fields.nx', type: 'number', default: 160, min: 20 },
      { name: 'ny', labelKey: 'production.fields.ny', type: 'number', default: 80, min: 10 },
      { name: 'medium', labelKey: 'production.fields.mediumType', type: 'select', default: 'random_cylinders', options: ['random_cylinders', 'tube_array'] },
      { name: 'model', labelKey: 'production.fields.multiphaseModel', type: 'select', default: 'cg', options: ['sc', 'cg'] },
      { name: 'porosity', labelKey: 'production.fields.porosity', type: 'number', default: 0.6, step: 0.05, min: 0.1, max: 0.95 },
      { name: 'n_steps', labelKey: 'production.fields.nSteps', type: 'number', default: 5000, min: 1 },
      { name: 'output_interval', labelKey: 'production.fields.outputInterval', type: 'number', default: 1000, min: 1 },
      { name: 'device', labelKey: 'production.fields.device', type: 'select', default: 'cpu', options: ['cpu', 'cuda:0'] },
      { name: 'seed', labelKey: 'production.fields.seed', type: 'number', default: 0, min: 0 },
    ],
  },
}

// ---------------------------------------------------------------------------
// 状态与表单
// ---------------------------------------------------------------------------

const router = useRouter()
const selectedType = ref<string>('cylinder_flow')
const formData = reactive<Record<string, any>>({})

const currentDef = computed<SolverTypeDef>(() => solverTypes[selectedType.value])
const currentFields = computed<SolverField[]>(() => currentDef.value.fields)
const currentDesc = computed(() => t(currentDef.value.descKey))

const submitting = ref(false)
const preflightLoading = ref(false)
const preflightResult = ref<Record<string, any> | null>(null)

const preflightChecks = computed(() => (preflightResult.value?.checks as any[]) || [])
const preflightTitle = computed(() => {
  const r = preflightResult.value
  if (!r) return ''
  const errors = (r.errors as any[]) || []
  const warnings = (r.warnings as any[]) || []
  if (errors.length) return t('production.solve.preflightErrors', { count: errors.length })
  if (warnings.length) return t('production.solve.preflightWarnings', { count: warnings.length })
  return t('production.solve.preflightPassed')
})
const preflightType = computed<'success' | 'warning' | 'error' | 'info'>(() => {
  const r = preflightResult.value
  if (!r) return 'info'
  const errors = (r.errors as any[]) || []
  const warnings = (r.warnings as any[]) || []
  if (errors.length) return 'error'
  if (warnings.length) return 'warning'
  return 'success'
})

function onTypeChange() {
  // 用默认值重建表单
  for (const f of currentFields.value) {
    formData[f.name] = f.default
  }
  // 清理旧字段
  for (const k of Object.keys(formData)) {
    if (!currentFields.value.some((f) => f.name === k)) {
      delete formData[k]
    }
  }
}

// 作业监控
const jobs = ref<Job[]>([])
const jobsLoading = ref(false)
const activeJob = ref<Job | null>(null)
const activeJobId = ref('')
const lastStep = ref(0)
const totalSteps = ref(0)

let pollTimer: ReturnType<typeof setInterval> | null = null
let listTimer: ReturnType<typeof setInterval> | null = null

const progressPercent = computed(() => {
  if (!totalSteps.value) return 0
  return Math.min(100, Math.round((lastStep.value / totalSteps.value) * 100))
})

function statusLabel(status: string): string {
  const map: Record<string, string> = {
    queued: t('production.status.queued'),
    running: t('production.status.running'),
    completed: t('production.status.completed'),
    failed: t('production.status.failed'),
    cancelled: t('production.status.cancelled'),
  }
  return map[status] || status
}

function statusTagType(status: string): 'success' | 'warning' | 'danger' | 'info' | 'primary' {
  const map: Record<string, 'success' | 'warning' | 'danger' | 'info' | 'primary'> = {
    queued: 'info',
    running: 'warning',
    completed: 'success',
    failed: 'danger',
    cancelled: 'info',
  }
  return map[status] || 'info'
}

function statusAlertType(status: string): 'success' | 'warning' | 'error' | 'info' {
  const map: Record<string, 'success' | 'warning' | 'error' | 'info'> = {
    queued: 'info',
    running: 'warning',
    completed: 'success',
    failed: 'error',
    cancelled: 'info',
  }
  return map[status] || 'info'
}

function formatTime(iso: string | null): string {
  if (!iso) return '—'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString()
}

function buildPayload(): Record<string, any> {
  const payload: Record<string, any> = {}
  for (const f of currentFields.value) {
    const v = formData[f.name]
    if (v !== undefined && v !== null && v !== '') {
      payload[f.name] = f.type === 'number' ? Number(v) : v
    }
  }
  return payload
}

async function submitJob() {
  submitting.value = true
  try {
    const payload = buildPayload()
    const res = await submitSolverJob(currentDef.value.endpoint, payload)
    activeJobId.value = res.job_id
    lastStep.value = 0
    totalSteps.value = Number(payload.n_steps) || 0
    ElMessage.success(t('production.solve.jobSubmittedMsg', { id: res.job_id }))
    await refreshActiveJob()
    startPolling()
    await loadJobs()
  } finally {
    submitting.value = false
  }
}

async function runPreflight() {
  preflightLoading.value = true
  try {
    const payload: Record<string, any> = { solver_type: selectedType.value }
    for (const f of currentFields.value) {
      const v = formData[f.name]
      if (v !== undefined && v !== null && v !== '') {
        payload[f.name] = f.type === 'number' ? Number(v) : v
      }
    }
    preflightResult.value = await preprocessPreflight(payload)
  } finally {
    preflightLoading.value = false
  }
}

async function refreshActiveJob() {
  if (!activeJobId.value) return
  try {
    const job = await getJob(activeJobId.value)
    activeJob.value = job
    totalSteps.value = Number((job.config?.n_steps as any) || totalSteps.value)
    // 拉取实时诊断数据（力系数/残差）用于进度
    try {
      const metrics = await getLiveMetrics(activeJobId.value, lastStep.value)
      if (metrics.diagnostics.length) {
        const last = metrics.diagnostics[metrics.diagnostics.length - 1]
        const step = Number(last.step ?? last.t ?? last.iter ?? 0)
        if (step > lastStep.value) lastStep.value = step
      }
    } catch {
      // 某些作业没有诊断数据，忽略
    }
    if (['completed', 'failed', 'cancelled'].includes(job.status)) {
      stopPolling()
    }
  } catch {
    // 作业查询失败时停止轮询
    stopPolling()
  }
}

function startPolling() {
  stopPolling()
  pollTimer = setInterval(() => {
    refreshActiveJob()
  }, 2000)
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

async function loadJobs() {
  jobsLoading.value = true
  try {
    const res = await listJobs({ limit: 0 })
    jobs.value = res.jobs
  } finally {
    jobsLoading.value = false
  }
}

async function cancel(row: Job) {
  try {
    await ElMessageBox.confirm(t('production.common.confirmCancel', { name: row.name }), t('production.common.hint'), {
      type: 'warning',
    })
  } catch {
    return
  }
  await cancelJob(row.job_id)
  ElMessage.success(t('production.common.cancelSent'))
  await loadJobs()
}

function goPostprocess(row: Job) {
  router.push({ path: '/production/postprocess', query: { job_id: row.job_id } })
}

onMounted(() => {
  onTypeChange()
  loadJobs()
  listTimer = setInterval(() => {
    if (!activeJobId.value) loadJobs()
  }, 8000)
})

onBeforeUnmount(() => {
  stopPolling()
  if (listTimer) {
    clearInterval(listTimer)
    listTimer = null
  }
})
</script>

<style scoped>
.desc-alert {
  margin-bottom: 12px;
}
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.active-job-alert {
  margin-bottom: 12px;
}
.progress-bar {
  margin-top: 8px;
}
.error-text {
  margin-top: 8px;
  color: #f56c6c;
  white-space: pre-wrap;
  font-family: monospace;
  font-size: 12px;
}
.preflight-list {
  margin: 0;
  padding-left: 16px;
}
.muted {
  color: #909399;
  font-size: 13px;
}
</style>
