<template>
  <div class="projects-view">
    <!-- ============================ 项目列表 ============================ -->
    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ t('misc.projects.title') }}</span>
          <div class="header-actions">
            <el-button :icon="Refresh" :loading="loadingProjects" @click="loadProjects">{{ t('misc.projects.refresh') }}</el-button>
            <el-button type="primary" :icon="Plus" @click="openProjectDialog()">{{ t('misc.projects.newProject') }}</el-button>
          </div>
        </div>
      </template>

      <el-table
        v-loading="loadingProjects"
        :data="projects"
        stripe
        border
        highlight-current-row
        style="width: 100%"
        @current-change="onSelectProject"
      >
        <el-table-column prop="name" :label="t('misc.projects.columns.name')" min-width="160" show-overflow-tooltip />
        <el-table-column prop="description" :label="t('misc.projects.columns.description')" min-width="220" show-overflow-tooltip />
        <el-table-column prop="owner" :label="t('misc.projects.columns.owner')" width="120" show-overflow-tooltip />
        <el-table-column :label="t('misc.projects.columns.tags')" min-width="150">
          <template #default="{ row }">
            <el-tag v-for="t in row.tags" :key="t" size="small" class="tag-margin">{{ t }}</el-tag>
            <span v-if="!row.tags || !row.tags.length" class="muted">—</span>
          </template>
        </el-table-column>
        <el-table-column :label="t('misc.projects.columns.createdAt')" width="175">
          <template #default="{ row }">{{ formatTime(row.created_at) }}</template>
        </el-table-column>
        <el-table-column :label="t('misc.projects.columns.actions')" width="210" align="center" fixed="right">
          <template #default="{ row }">
            <el-button size="small" type="primary" :icon="View" @click.stop="selectProject(row)">{{ t('misc.projects.cases') }}</el-button>
            <el-button size="small" :icon="Edit" @click.stop="openProjectDialog(row)">{{ t('misc.projects.edit') }}</el-button>
            <el-button size="small" type="danger" :icon="Delete" @click.stop="onDeleteProject(row)">{{ t('misc.projects.delete') }}</el-button>
          </template>
        </el-table-column>
        <template #empty>
          <el-empty :description="t('misc.projects.empty')" />
        </template>
      </el-table>
    </el-card>

    <!-- ============================ 案例列表 ============================ -->
    <el-card v-if="selectedProject" class="cases-card">
      <template #header>
        <div class="card-header">
          <span>
            {{ t('misc.projects.caseManagement') }} — <strong>{{ selectedProject.name }}</strong>
            <span class="muted">（{{ selectedProject.id }}）</span>
          </span>
          <div class="header-actions">
            <el-button :icon="Refresh" :loading="loadingCases" @click="loadCases">{{ t('misc.projects.refresh') }}</el-button>
            <el-button type="primary" :icon="Plus" @click="openCaseDialog()">{{ t('misc.projects.newCase') }}</el-button>
          </div>
        </div>
      </template>

      <el-table
        v-loading="loadingCases"
        :data="cases"
        stripe
        border
        size="small"
        style="width: 100%"
      >
        <el-table-column prop="name" :label="t('misc.projects.caseColumns.name')" min-width="170" show-overflow-tooltip />
        <el-table-column prop="scenario" :label="t('misc.projects.caseColumns.scenario')" width="110" />
        <el-table-column :label="t('misc.projects.caseColumns.status')" width="100">
          <template #default="{ row }">
            <el-tag :type="statusTagType(row.status)" disable-transitions>{{ row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column :label="t('misc.projects.caseColumns.workflowStage')" width="150">
          <template #default="{ row }">
            <el-tag :type="stageTagType(row.workflow_stage)" disable-transitions>
              {{ stageLabel(row.workflow_stage) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column :label="t('misc.projects.caseColumns.config')" width="90" align="center">
          <template #default="{ row }">
            <span class="config-hint" :title="JSON.stringify(row.config)">{{ t('misc.projects.configItemCount', { count: configKeyCount(row.config) }) }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="job_id" :label="t('misc.projects.caseColumns.jobId')" width="110" show-overflow-tooltip>
          <template #default="{ row }">
            <span v-if="row.job_id">{{ row.job_id }}</span>
            <span v-else class="muted">—</span>
          </template>
        </el-table-column>
        <el-table-column :label="t('misc.projects.caseColumns.actions')" width="290" align="center" fixed="right">
          <template #default="{ row }">
            <el-button size="small" :icon="Edit" @click="openCaseDialog(row)">{{ t('misc.projects.edit') }}</el-button>
            <el-button size="small" :icon="DocumentCopy" @click="onCloneCase(row)">{{ t('misc.projects.clone') }}</el-button>
            <el-button
              size="small"
              type="primary"
              :icon="CaretRight"
              :disabled="isFinalStage(row.workflow_stage)"
              @click="onAdvanceWorkflow(row)"
            >
              {{ t('misc.projects.advance') }}
            </el-button>
            <el-button size="small" type="danger" :icon="Delete" @click="onDeleteCase(row)">{{ t('misc.projects.delete') }}</el-button>
          </template>
        </el-table-column>
        <template #empty>
          <el-empty :description="t('misc.projects.emptyCases')" :image-size="80" />
        </template>
      </el-table>
    </el-card>

    <!-- ============================ 项目新建/编辑弹窗 ============================ -->
    <el-dialog
      v-model="projectDialogVisible"
      :title="editingProject ? t('misc.projects.editProject') : t('misc.projects.newProject')"
      width="520px"
      destroy-on-close
    >
      <el-form ref="projectFormRef" :model="projectForm" :rules="projectRules" label-width="80px">
        <el-form-item :label="t('misc.projects.form.name')" prop="name">
          <el-input v-model="projectForm.name" :placeholder="t('misc.projects.form.namePlaceholder')" maxlength="120" />
        </el-form-item>
        <el-form-item :label="t('misc.projects.form.description')">
          <el-input
            v-model="projectForm.description"
            type="textarea"
            :rows="3"
            :placeholder="t('misc.projects.form.descriptionPlaceholder')"
            maxlength="1000"
          />
        </el-form-item>
        <el-form-item :label="t('misc.projects.form.owner')">
          <el-input v-model="projectForm.owner" :placeholder="t('misc.projects.form.ownerPlaceholder')" maxlength="120" />
        </el-form-item>
        <el-form-item :label="t('misc.projects.form.tags')">
          <el-input v-model="projectForm.tagsText" :placeholder="t('misc.projects.form.tagsPlaceholder')" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="projectDialogVisible = false">{{ t('misc.projects.cancel') }}</el-button>
        <el-button type="primary" :loading="savingProject" @click="submitProject">{{ t('misc.projects.save') }}</el-button>
      </template>
    </el-dialog>

    <!-- ============================ 案例新建/编辑弹窗 ============================ -->
    <el-dialog
      v-model="caseDialogVisible"
      :title="editingCase ? t('misc.projects.editCase') : t('misc.projects.newCase')"
      width="560px"
      destroy-on-close
    >
      <el-form ref="caseFormRef" :model="caseForm" :rules="caseRules" label-width="100px">
        <el-form-item :label="t('misc.projects.form.name')" prop="name">
          <el-input v-model="caseForm.name" :placeholder="t('misc.projects.form.caseNamePlaceholder')" maxlength="120" />
        </el-form-item>
        <el-form-item :label="t('misc.projects.form.description')">
          <el-input
            v-model="caseForm.description"
            type="textarea"
            :rows="2"
            :placeholder="t('misc.projects.form.caseDescriptionPlaceholder')"
            maxlength="1000"
          />
        </el-form-item>
        <el-form-item :label="t('misc.projects.form.scenario')">
          <el-input v-model="caseForm.scenario" :placeholder="t('misc.projects.form.scenarioPlaceholder')" maxlength="80" />
        </el-form-item>
        <el-form-item :label="t('misc.projects.form.workflowStage')">
          <el-select v-model="caseForm.workflow_stage" style="width: 100%">
            <el-option
              v-for="s in WORKFLOW_STAGES"
              :key="s"
              :label="stageLabel(s)"
              :value="s"
            />
          </el-select>
        </el-form-item>
        <el-form-item v-if="editingCase" :label="t('misc.projects.form.status')">
          <el-input v-model="caseForm.status" :placeholder="t('misc.projects.form.statusPlaceholder')" maxlength="40" />
        </el-form-item>
        <el-form-item v-if="editingCase" :label="t('misc.projects.form.jobId')">
          <el-input v-model="caseForm.job_id" :placeholder="t('misc.projects.form.jobIdPlaceholder')" />
        </el-form-item>
        <el-form-item :label="t('misc.projects.form.configJson')" prop="configText">
          <el-input
            v-model="caseForm.configText"
            type="textarea"
            :rows="5"
            placeholder='{ "nx": 160, "ny": 60 }'
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="caseDialogVisible = false">{{ t('misc.projects.cancel') }}</el-button>
        <el-button type="primary" :loading="savingCase" @click="submitCase">{{ t('misc.projects.save') }}</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { onMounted, reactive, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { ElMessage, ElMessageBox } from 'element-plus'
import type { FormInstance, FormRules } from 'element-plus'
import {
  CaretRight,
  Delete,
  DocumentCopy,
  Edit,
  Plus,
  Refresh,
  View,
} from '@element-plus/icons-vue'
import {
  WORKFLOW_STAGES,
  advanceWorkflow,
  cloneCase,
  createCase,
  createProject,
  deleteCase,
  deleteProject,
  listCases,
  listProjects,
  updateCase,
  updateProject,
  type Project,
  type SimulationCase,
} from '@/api/projects'

const { t } = useI18n()

// ---------------------------------------------------------------------------
// 工作流阶段 / 状态的展示映射
// ---------------------------------------------------------------------------

const STAGE_KEYS: Record<string, string> = {
  draft: 'misc.projects.stages.draft',
  setup: 'misc.projects.stages.setup',
  meshed: 'misc.projects.stages.meshed',
  solved: 'misc.projects.stages.solved',
  post_processed: 'misc.projects.stages.postProcessed',
}

function stageLabel(stage: string): string {
  const key = STAGE_KEYS[stage]
  return key ? t(key) : stage
}

function stageTagType(stage: string): 'info' | 'warning' | 'success' | 'primary' {
  const map: Record<string, 'info' | 'warning' | 'success' | 'primary'> = {
    draft: 'info',
    setup: 'warning',
    meshed: 'warning',
    solved: 'success',
    post_processed: 'success',
  }
  return map[stage] || 'info'
}

function statusTagType(status: string): 'info' | 'warning' | 'success' | 'danger' | 'primary' {
  const map: Record<string, 'info' | 'warning' | 'success' | 'danger' | 'primary'> = {
    draft: 'info',
    solved: 'success',
    running: 'warning',
    failed: 'danger',
    completed: 'success',
  }
  return map[status] || 'primary'
}

function isFinalStage(stage: string): boolean {
  return stage === WORKFLOW_STAGES[WORKFLOW_STAGES.length - 1]
}

function configKeyCount(config: Record<string, any> | null | undefined): number {
  if (!config || typeof config !== 'object') return 0
  return Object.keys(config).length
}

function formatTime(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString()
}

function parseTags(text: string): string[] {
  return text
    .split(/[,，;；\s]+/)
    .map((t) => t.trim())
    .filter(Boolean)
}

// ---------------------------------------------------------------------------
// 项目列表状态
// ---------------------------------------------------------------------------

const loadingProjects = ref(false)
const projects = ref<Project[]>([])
const selectedProject = ref<Project | null>(null)

async function loadProjects() {
  loadingProjects.value = true
  try {
    projects.value = await listProjects()
  } catch {
    // 错误提示由 request 拦截器统一处理
  } finally {
    loadingProjects.value = false
  }
}

function onSelectProject(row: Project | null) {
  selectedProject.value = row
  if (row) loadCases()
}

function selectProject(row: Project) {
  selectedProject.value = row
  loadCases()
}

// ---------------------------------------------------------------------------
// 项目新建/编辑
// ---------------------------------------------------------------------------

const projectDialogVisible = ref(false)
const savingProject = ref(false)
const editingProject = ref<Project | null>(null)
const projectFormRef = ref<FormInstance>()
const projectForm = reactive({ name: '', description: '', owner: '', tagsText: '' })
const projectRules: FormRules = {
  name: [{ required: true, message: t('misc.projects.messages.enterProjectName'), trigger: 'blur' }],
}

function openProjectDialog(project?: Project) {
  editingProject.value = project ?? null
  projectForm.name = project?.name ?? ''
  projectForm.description = project?.description ?? ''
  projectForm.owner = project?.owner ?? ''
  projectForm.tagsText = (project?.tags ?? []).join(', ')
  projectDialogVisible.value = true
}

async function submitProject() {
  if (!projectFormRef.value) return
  const ok = await projectFormRef.value.validate().catch(() => false)
  if (!ok) return

  savingProject.value = true
  try {
    const payload = {
      name: projectForm.name.trim(),
      description: projectForm.description.trim(),
      owner: projectForm.owner.trim(),
      tags: parseTags(projectForm.tagsText),
    }
    if (editingProject.value) {
      await updateProject(editingProject.value.id, payload)
      ElMessage.success(t('misc.projects.messages.projectUpdated'))
    } else {
      await createProject(payload)
      ElMessage.success(t('misc.projects.messages.projectCreated'))
    }
    projectDialogVisible.value = false
    await loadProjects()
  } catch {
    // 错误提示由拦截器处理
  } finally {
    savingProject.value = false
  }
}

async function onDeleteProject(project: Project) {
  try {
    await ElMessageBox.confirm(
      t('misc.projects.messages.confirmDeleteProject', { name: project.name }),
      t('misc.projects.messages.deleteConfirmTitle'),
      {
        type: 'warning',
        confirmButtonText: t('misc.projects.delete'),
        cancelButtonText: t('misc.projects.cancel'),
      },
    )
  } catch {
    return
  }
  try {
    await deleteProject(project.id)
    ElMessage.success(t('misc.projects.messages.projectDeleted'))
    if (selectedProject.value?.id === project.id) {
      selectedProject.value = null
      cases.value = []
    }
    await loadProjects()
  } catch {
    // 错误提示由拦截器处理
  }
}

// ---------------------------------------------------------------------------
// 案例列表状态
// ---------------------------------------------------------------------------

const loadingCases = ref(false)
const cases = ref<SimulationCase[]>([])

async function loadCases() {
  if (!selectedProject.value) return
  loadingCases.value = true
  try {
    cases.value = await listCases(selectedProject.value.id)
  } catch {
    cases.value = []
  } finally {
    loadingCases.value = false
  }
}

// ---------------------------------------------------------------------------
// 案例新建/编辑
// ---------------------------------------------------------------------------

const caseDialogVisible = ref(false)
const savingCase = ref(false)
const editingCase = ref<SimulationCase | null>(null)
const caseFormRef = ref<FormInstance>()
const caseForm = reactive({
  name: '',
  description: '',
  scenario: 'custom',
  workflow_stage: 'draft',
  status: '',
  job_id: '',
  configText: '{}',
})
const caseRules: FormRules = {
  name: [{ required: true, message: t('misc.projects.messages.enterCaseName'), trigger: 'blur' }],
  configText: [
    {
      validator: (_rule, value: string, callback) => {
        try {
          JSON.parse(value || '{}')
          callback()
        } catch {
          callback(new Error(t('misc.projects.messages.invalidConfig')))
        }
      },
      trigger: 'blur',
    },
  ],
}

function openCaseDialog(c?: SimulationCase) {
  editingCase.value = c ?? null
  caseForm.name = c?.name ?? ''
  caseForm.description = c?.description ?? ''
  caseForm.scenario = c?.scenario ?? 'custom'
  caseForm.workflow_stage = c?.workflow_stage ?? 'draft'
  caseForm.status = c?.status ?? ''
  caseForm.job_id = c?.job_id ?? ''
  caseForm.configText = c?.config ? JSON.stringify(c.config, null, 2) : '{}'
  caseDialogVisible.value = true
}

function parseConfigText(text: string): Record<string, any> | null {
  try {
    const parsed = JSON.parse(text.trim() || '{}')
    return parsed && typeof parsed === 'object' ? parsed : null
  } catch {
    return null
  }
}

async function submitCase() {
  if (!selectedProject.value) return
  if (!caseFormRef.value) return
  const ok = await caseFormRef.value.validate().catch(() => false)
  if (!ok) return

  const config = parseConfigText(caseForm.configText)
  if (config === null) {
    ElMessage.warning(t('misc.projects.messages.invalidConfigObject'))
    return
  }

  savingCase.value = true
  try {
    if (editingCase.value) {
      await updateCase(selectedProject.value.id, editingCase.value.id, {
        name: caseForm.name.trim(),
        description: caseForm.description.trim(),
        scenario: caseForm.scenario.trim(),
        status: caseForm.status.trim(),
        workflow_stage: caseForm.workflow_stage,
        config,
        job_id: caseForm.job_id.trim() || null,
      })
      ElMessage.success(t('misc.projects.messages.caseUpdated'))
    } else {
      await createCase(selectedProject.value.id, {
        name: caseForm.name.trim(),
        description: caseForm.description.trim(),
        scenario: caseForm.scenario.trim(),
        workflow_stage: caseForm.workflow_stage,
        config,
      })
      ElMessage.success(t('misc.projects.messages.caseCreated'))
    }
    caseDialogVisible.value = false
    await loadCases()
  } catch {
    // 错误提示由拦截器处理
  } finally {
    savingCase.value = false
  }
}

async function onDeleteCase(c: SimulationCase) {
  if (!selectedProject.value) return
  try {
    await ElMessageBox.confirm(t('misc.projects.messages.confirmDeleteCase', { name: c.name }), t('misc.projects.messages.deleteConfirmTitle'), {
      type: 'warning',
      confirmButtonText: t('misc.projects.delete'),
      cancelButtonText: t('misc.projects.cancel'),
    })
  } catch {
    return
  }
  try {
    await deleteCase(selectedProject.value.id, c.id)
    ElMessage.success(t('misc.projects.messages.caseDeleted'))
    await loadCases()
  } catch {
    // 错误提示由拦截器处理
  }
}

async function onCloneCase(c: SimulationCase) {
  if (!selectedProject.value) return
  let newName = ''
  try {
    const res = await ElMessageBox.prompt(t('misc.projects.messages.clonePrompt'), t('misc.projects.messages.cloneTitle', { name: c.name }), {
      confirmButtonText: t('misc.projects.clone'),
      cancelButtonText: t('misc.projects.cancel'),
      inputValue: '',
      inputPlaceholder: `${c.name} (copy)`,
    })
    newName = (res.value || '').trim()
  } catch {
    return
  }
  try {
    await cloneCase(selectedProject.value.id, c.id, { name: newName || undefined })
    ElMessage.success(t('misc.projects.messages.caseCloned'))
    await loadCases()
  } catch {
    // 错误提示由拦截器处理
  }
}

async function onAdvanceWorkflow(c: SimulationCase) {
  if (!selectedProject.value) return
  try {
    const updated = await advanceWorkflow(selectedProject.value.id, c.id)
    ElMessage.success(t('misc.projects.messages.workflowAdvanced', { stage: stageLabel(updated.workflow_stage) }))
    await loadCases()
  } catch {
    // 错误提示由拦截器处理（如已处于最终阶段 409）
  }
}

onMounted(loadProjects)
</script>

<style scoped>
.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.header-actions {
  display: flex;
  gap: 8px;
}
.cases-card {
  margin-top: 16px;
}
.tag-margin {
  margin: 0 4px 4px 0;
}
.muted {
  color: #909399;
  font-size: 13px;
}
.config-hint {
  cursor: help;
  border-bottom: 1px dashed #c0c4cc;
}
</style>
