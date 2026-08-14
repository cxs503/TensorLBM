<template>
  <div>
    <el-card>
      <template #header>
        <div class="card-header">
          <span>CAD 船体建模</span>
          <el-select
            v-model="hullType"
            placeholder="选择船型"
            style="width: 260px"
          >
            <el-option
              v-for="h in hullTypes"
              :key="h.value"
              :label="h.label"
              :value="h.value"
            />
          </el-select>
        </div>
      </template>

      <el-alert
        v-if="currentHullDesc"
        :title="currentHullDesc"
        type="info"
        :closable="false"
        class="desc-alert"
      />

      <el-form :model="form" label-width="120px" class="param-form">
        <el-row :gutter="16">
          <el-col :span="8">
            <el-form-item label="船长 L (lu)">
              <el-input-number v-model="form.length" :min="20" :step="10" controls-position="right" style="width: 100%" />
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="船宽 B (lu)">
              <el-input-number v-model="form.beam" :min="4" :step="1" controls-position="right" style="width: 100%" />
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="吃水 T (lu)">
              <el-input-number v-model="form.draft" :min="2" :step="1" controls-position="right" style="width: 100%" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="16">
          <el-col :span="8">
            <el-form-item label="剖面站数">
              <el-input-number v-model="form.n_stations" :min="3" :max="41" controls-position="right" style="width: 100%" />
            </el-form-item>
          </el-col>
          <el-col :span="16">
            <el-form-item label=" ">
              <el-button type="primary" :icon="View" :loading="previewLoading" @click="generatePreview">
                生成三视图预览
              </el-button>
              <el-button :icon="Download" :loading="exportLoading" @click="exportStl">导出 STL</el-button>
            </el-form-item>
          </el-col>
        </el-row>
      </el-form>
    </el-card>

    <el-row :gutter="16" class="content-row">
      <el-col :span="12">
        <el-card>
          <template #header><span>船体三视图</span></template>
          <div v-loading="previewLoading" class="preview-body">
            <img v-if="previewImage" :src="previewImage" alt="preview" class="result-img" />
            <el-empty v-else description="点击「生成三视图预览」" :image-size="80" />
          </div>
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card>
          <template #header><span>船型统计参数</span></template>
          <el-table v-if="previewStats" :data="statsRows" size="small" border>
            <el-table-column prop="label" label="参数" width="160" />
            <el-table-column prop="value" label="数值" />
          </el-table>
          <el-empty v-else description="暂无统计数据" :image-size="80" />
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="16" class="content-row">
      <el-col :span="12">
        <el-card>
          <template #header><span>体素掩码（3D 网格）</span></template>
          <el-form :model="maskForm" label-width="90px" size="small">
            <el-row :gutter="8">
              <el-col :span="8">
                <el-form-item label="nx"><el-input-number v-model="maskForm.nx" :min="20" controls-position="right" style="width: 100%" /></el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item label="ny"><el-input-number v-model="maskForm.ny" :min="10" controls-position="right" style="width: 100%" /></el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item label="nz"><el-input-number v-model="maskForm.nz" :min="10" controls-position="right" style="width: 100%" /></el-form-item>
              </el-col>
            </el-row>
            <el-button type="primary" size="small" :loading="maskLoading" @click="generateMask">生成体素掩码</el-button>
          </el-form>
          <div v-loading="maskLoading" class="mask-body">
            <img v-if="maskImage" :src="maskImage" alt="mask" class="result-img" />
            <el-empty v-else description="点击「生成体素掩码」" :image-size="60" />
          </div>
          <el-descriptions v-if="maskStats" :column="2" size="small" border class="mask-stats">
            <el-descriptions-item label="Cb（数值）">{{ maskStats.Cb_numerical }}</el-descriptions-item>
            <el-descriptions-item label="固体网格">{{ maskStats.solid_cells }}</el-descriptions-item>
            <el-descriptions-item label="流体网格">{{ maskStats.fluid_cells }}</el-descriptions-item>
            <el-descriptions-item label="网格">{{ maskStats.nx }}×{{ maskStats.ny }}×{{ maskStats.nz }}</el-descriptions-item>
          </el-descriptions>
        </el-card>
      </el-col>

      <el-col :span="12">
        <el-card>
          <template #header><span>LBM 无量纲参数换算</span></template>
          <el-form :model="lbmForm" label-width="120px" size="small">
            <el-form-item label="船长 (m)"><el-input-number v-model="lbmForm.length_m" :min="1" controls-position="right" style="width: 100%" /></el-form-item>
            <el-form-item label="航速 (m/s)"><el-input-number v-model="lbmForm.speed_ms" :min="0.1" :step="0.5" controls-position="right" style="width: 100%" /></el-form-item>
            <el-form-item label="运动粘度 (m²/s)">
              <el-input-number v-model="lbmForm.nu_m2s" :min="1e-8" :step="1e-7" :precision="10" controls-position="right" style="width: 100%" />
            </el-form-item>
            <el-button type="primary" size="small" :loading="lbmLoading" @click="computeLbm">计算 LBM 参数</el-button>
          </el-form>
          <el-table v-if="lbmResult" :data="lbmRows" size="small" border class="lbm-table">
            <el-table-column prop="label" label="参数" width="140" />
            <el-table-column prop="value" label="数值" />
          </el-table>
        </el-card>
      </el-col>
    </el-row>

    <el-card class="content-row">
      <template #header><span>直接提交求解器（CAD → Solver）</span></template>
      <el-form :model="solverForm" label-width="150px" size="small">
        <el-row :gutter="12">
          <el-col :span="6">
            <el-form-item label="网格 nx"><el-input-number v-model="solverForm.nx" :min="20" controls-position="right" style="width: 100%" /></el-form-item>
          </el-col>
          <el-col :span="6">
            <el-form-item label="网格 ny"><el-input-number v-model="solverForm.ny" :min="10" controls-position="right" style="width: 100%" /></el-form-item>
          </el-col>
          <el-col :span="6">
            <el-form-item label="网格 nz"><el-input-number v-model="solverForm.nz" :min="10" controls-position="right" style="width: 100%" /></el-form-item>
          </el-col>
          <el-col :span="6">
            <el-form-item label="雷诺数 Re"><el-input-number v-model="solverForm.re" :min="1" controls-position="right" style="width: 100%" /></el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="12">
          <el-col :span="6">
            <el-form-item label="入流速度 u_in"><el-input-number v-model="solverForm.u_in" :min="0.001" :step="0.01" controls-position="right" style="width: 100%" /></el-form-item>
          </el-col>
          <el-col :span="6">
            <el-form-item label="时间步数"><el-input-number v-model="solverForm.n_steps" :min="1" :step="100" controls-position="right" style="width: 100%" /></el-form-item>
          </el-col>
          <el-col :span="6">
            <el-form-item label="输出间隔"><el-input-number v-model="solverForm.output_interval" :min="1" :step="50" controls-position="right" style="width: 100%" /></el-form-item>
          </el-col>
          <el-col :span="6">
            <el-form-item label="计算设备">
              <el-select v-model="solverForm.device" style="width: 100%">
                <el-option label="cpu" value="cpu" />
                <el-option label="cuda:0" value="cuda:0" />
              </el-select>
            </el-form-item>
          </el-col>
        </el-row>
        <el-button type="primary" :loading="solverLoading" @click="sendToSolver">提交求解作业</el-button>
        <el-tag v-if="solverJobId" type="success" class="job-tag">
          已提交作业：{{ solverJobId }}
          <el-button link type="primary" @click="$router.push('/production/dashboard')">前往总览</el-button>
        </el-tag>
      </el-form>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { Download, View } from '@element-plus/icons-vue'
import {
  cadHullMask,
  cadHullTypes,
  cadLbmParameters,
  cadPreview,
  cadSendToSolver,
  type HullTypeItem,
} from '@/api/production'

const hullTypes = ref<HullTypeItem[]>([])
const hullType = ref('series60')
const currentHullDesc = computed(() => {
  const h = hullTypes.value.find((x) => x.value === hullType.value)
  return h ? `${h.label}：${h.description}` : ''
})

// 三视图预览
const form = reactive({ length: 100, beam: 16, draft: 8, n_stations: 11 })
const previewLoading = ref(false)
const previewImage = ref('')
const previewStats = ref<Record<string, any> | null>(null)

const statsRows = computed(() => {
  const s = previewStats.value
  if (!s) return []
  const rows: { label: string; value: string }[] = [
    { label: '船型', value: String(s.label ?? '') },
    { label: 'Cb', value: String(s.Cb) },
    { label: 'Cwp', value: String(s.Cwp) },
    { label: 'Cm', value: String(s.Cm) },
    { label: 'Cp', value: String(s.Cp) },
    { label: 'L/B', value: String(s['L/B']) },
    { label: 'B/T', value: String(s['B/T']) },
    { label: '排水量 (lu³)', value: String(s.displacement_lu3) },
  ]
  return rows
})

// 体素掩码
const maskForm = reactive({ nx: 160, ny: 60, nz: 40 })
const maskLoading = ref(false)
const maskImage = ref('')
const maskStats = ref<Record<string, any> | null>(null)

// LBM 参数
const lbmForm = reactive({
  length_m: 100,
  speed_ms: 5,
  nu_m2s: 1.139e-6,
  lbm_length: 100,
  lbm_speed: 0.05,
  froude_target: null as number | null,
})
const lbmLoading = ref(false)
const lbmResult = ref<Record<string, any> | null>(null)

const lbmRows = computed(() => {
  const r = lbmResult.value
  if (!r) return []
  return [
    { label: 'Re', value: String(r.re_physical) },
    { label: 'Fr', value: String(r.froude_number) },
    { label: 'dx (m)', value: String(r.dx_m) },
    { label: 'dt (s)', value: String(r.dt_s) },
    { label: 'τ', value: `${r.lbm_tau} ${r.stable ? '（稳定）' : '（不稳定）'}` },
    { label: 'Ma', value: String(r.mach_number) },
  ]
})

// 直接提交求解
const solverForm = reactive({
  nx: 160,
  ny: 60,
  nz: 40,
  hull_length: 80,
  hull_beam: 8,
  hull_draft: 12,
  u_in: 0.05,
  re: 200,
  smagorinsky_cs: 0.1,
  n_steps: 2000,
  output_interval: 200,
  device: 'cpu',
})
const solverLoading = ref(false)
const solverJobId = ref('')
const exportLoading = ref(false)

async function loadHullTypes() {
  try {
    const res = await cadHullTypes()
    hullTypes.value = res.hull_types
  } catch {
    // 后端不可用时使用内置默认值
    hullTypes.value = [
      { value: 'wigley', label: 'Wigley Parabolic', description: 'ITTC 基准船型，抛物线横剖面', Cb: 0.4444 },
      { value: 'series60', label: 'Series 60 (Cb=0.60)', description: 'DTMB Series 60 标准商船船型', Cb: 0.6 },
      { value: 'kcs', label: 'KCS Approximation (Cb≈0.651)', description: 'KRISO 集装箱船近似船型', Cb: 0.651 },
    ]
  }
}

async function generatePreview() {
  previewLoading.value = true
  try {
    const r = await cadPreview({
      hull_type: hullType.value,
      length: form.length,
      beam: form.beam,
      draft: form.draft,
      n_stations: form.n_stations,
    })
    previewImage.value = r.image
    previewStats.value = r.stats
    // 同步求解参数
    solverForm.hull_length = form.length
    solverForm.hull_beam = form.beam
    solverForm.hull_draft = form.draft
  } finally {
    previewLoading.value = false
  }
}

async function generateMask() {
  maskLoading.value = true
  try {
    const r = await cadHullMask({
      hull_type: hullType.value,
      nx: maskForm.nx,
      ny: maskForm.ny,
      nz: maskForm.nz,
      length: form.length,
      beam: form.beam,
      draft: form.draft,
    })
    maskImage.value = r.image
    maskStats.value = r.stats
  } finally {
    maskLoading.value = false
  }
}

async function computeLbm() {
  lbmLoading.value = true
  try {
    lbmResult.value = await cadLbmParameters({
      length_m: lbmForm.length_m,
      speed_ms: lbmForm.speed_ms,
      nu_m2s: lbmForm.nu_m2s,
      lbm_length: lbmForm.lbm_length,
      lbm_speed: lbmForm.lbm_speed,
      froude_target: lbmForm.froude_target,
    })
  } finally {
    lbmLoading.value = false
  }
}

async function sendToSolver() {
  solverLoading.value = true
  try {
    const r = await cadSendToSolver({
      hull_type: hullType.value,
      ...solverForm,
      wave_amp: 0,
      wave_period: 200,
      seed: 0,
    })
    solverJobId.value = r.job_id
    ElMessage.success(`作业已提交：${r.job_id}`)
  } finally {
    solverLoading.value = false
  }
}

async function exportStl() {
  exportLoading.value = true
  try {
    const resp = await fetch('/api/cad/export-stl', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        hull_type: hullType.value,
        length: form.length,
        beam: form.beam,
        draft: form.draft,
        n_long: 60,
        n_vert: 30,
      }),
    })
    if (!resp.ok) throw new Error(await resp.text())
    const blob = await resp.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${hullType.value}_hull.stl`
    a.click()
    URL.revokeObjectURL(url)
    ElMessage.success('STL 已导出')
  } catch (e) {
    ElMessage.error(`STL 导出失败：${(e as Error).message}`)
  } finally {
    exportLoading.value = false
  }
}

onMounted(() => {
  loadHullTypes()
})
</script>

<style scoped>
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.desc-alert {
  margin-bottom: 16px;
}
.content-row {
  margin-top: 16px;
}
.preview-body,
.mask-body {
  min-height: 160px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.result-img {
  max-width: 100%;
  max-height: 420px;
  display: block;
  margin: 0 auto;
}
.mask-stats,
.lbm-table {
  margin-top: 12px;
}
.job-tag {
  margin-left: 12px;
}
</style>
