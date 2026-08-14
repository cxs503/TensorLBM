import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    redirect: '/production/dashboard',
  },
  {
    path: '/production',
    name: 'production',
    component: () => import('@/views/production/DashboardView.vue'),
    meta: { title: '数据生产', group: 'production' },
    children: [
      { path: 'dashboard', name: 'dashboard', component: () => import('@/views/production/DashboardView.vue'), meta: { title: '总览' } },
      { path: 'cad', name: 'cad', component: () => import('@/views/production/CadView.vue'), meta: { title: 'CAD 建模' } },
      { path: 'preprocess', name: 'preprocess', component: () => import('@/views/production/PreprocessView.vue'), meta: { title: '预处理' } },
      { path: 'solve', name: 'solve', component: () => import('@/views/production/SolveView.vue'), meta: { title: '求解' } },
      { path: 'postprocess', name: 'postprocess', component: () => import('@/views/production/PostprocessView.vue'), meta: { title: '后处理' } },
      { path: 'benchmarks', name: 'benchmarks', component: () => import('@/views/production/BenchmarksView.vue'), meta: { title: '基准' } },
    ],
  },
  {
    path: '/ai4s',
    name: 'ai4s',
    redirect: '/ai4s/apps',
    meta: { title: 'AI4S 应用', group: 'ai4s' },
    children: [
      { path: 'apps', name: 'apps', component: () => import('@/views/ai4s/AppsView.vue'), meta: { title: '应用列表' } },
      { path: 'run/:name?', name: 'app-run', component: () => import('@/views/ai4s/AppRunView.vue'), meta: { title: '运行应用' } },
      { path: 'lineage', name: 'app-lineage', component: () => import('@/views/ai4s/AppLineageView.vue'), meta: { title: '血缘追溯' } },
    ],
  },
  {
    path: '/data',
    name: 'data',
    component: () => import('@/views/data/DataCatalogView.vue'),
    meta: { title: '数据管理', group: 'data' },
  },
  {
    path: '/misc',
    name: 'misc',
    redirect: '/misc/projects',
    meta: { title: '项目', group: 'misc' },
    children: [
      { path: 'projects', name: 'projects', component: () => import('@/views/misc/ProjectsView.vue'), meta: { title: '项目' } },
      { path: 'reports', name: 'reports', component: () => import('@/views/misc/ReportsView.vue'), meta: { title: '报告' } },
    ],
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
