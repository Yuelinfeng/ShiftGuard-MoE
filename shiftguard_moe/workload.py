from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class PromptSpec:
    prompt_id: str
    domain: str
    text: str


@dataclass(slots=True, frozen=True)
class WorkloadItem:
    scenario: str
    window_index: int
    position: int
    prompt_id: str
    domain: str


DOMAIN_BLUEPRINTS: dict[str, dict[str, list[str]]] = {
    "code": {
        "bases": [
            "你是资深 Python 工程师。请分析下面这个异步服务在高并发下出现延迟抖动的根因，并给出最小修复方案：服务会并发拉取 24 个上游接口，再把结果聚合成 JSON 返回。",
            "阅读下面的 SQL 优化场景：一张 3 亿行订单表按用户和时间过滤，查询经常出现 Hash Join 回退。请给出索引、执行计划和改写建议。",
            "请像代码审查一样检查这个 Rust 微服务设计：它需要把 Kafka 消费到的消息写入 RocksDB，并在失败时支持幂等重试。哪些地方最容易埋下吞吐瓶颈？",
            "下面是一段多线程 C++ 推理服务的描述：请求线程会从共享队列拉任务，再调用 GPU worker。请推断可能出现的锁竞争和背压问题，并给出最小改造建议。",
            "请调试一个前端构建问题：Vite 项目在开发模式正常，但 production build 后某个动态 import 页面偶发白屏。你会如何定位 bundling 和 chunk preload 的问题？",
            "给你一个日志聚合组件的抽象设计：Fluent Bit 收集日志，经过 Lua filter 再进入 ClickHouse。请找出最可能导致数据乱序和重复写入的地方。",
        ],
        "modifiers": [
            "请把回答控制在 6 条以内，并优先保留最小可验证修复。",
            "如果你需要举例，请给出伪代码而不是完整工程。",
            "请特别关注延迟尾部而不是平均值。",
            "请把可能的观测指标也列出来，例如 queue depth、p99、retry rate。",
            "请说明哪些修改可能引入新的正确性风险。",
            "请先给出排查顺序，再给修复建议。",
        ],
    },
    "math": {
        "bases": [
            "请证明：如果一个序列满足 a_(n+1)=sqrt(2+a_n) 且 a_1=sqrt(2)，那么它单调递增并收敛。请给出严格证明。",
            "给定一个带约束的最优化问题：最小化 x^2+y^2，约束是 x+y=1 且 x>=0,y>=0。请用拉格朗日乘子法和几何解释各做一次。",
            "请分析一个概率问题：袋子里有 5 个红球、4 个蓝球、3 个绿球，不放回抽 4 次。求恰好出现 2 个红球的概率，并解释组合数来源。",
            "请解决一个离散数学题：证明任意树都满足边数等于节点数减一，并给出归纳法版本和图结构版本两个证明。",
            "考虑函数 f(x)=x^3-3x+1。请判断极值点、拐点，并说明牛顿法在不同初值下为什么可能收敛到不同根。",
            "请从线性代数角度解释为什么对称矩阵总能正交对角化，并补充一个 2x2 的具体例子。",
        ],
        "modifiers": [
            "请把每一步推导都写清楚，不要直接跳结论。",
            "请给出一个容易犯错的地方，并解释为什么错。",
            "如果存在多种做法，请明确比较它们的优缺点。",
            "请在结尾补一段直观解释，面向刚学过这章的学生。",
            "请尽量避免使用过于高级的定理。",
            "请在最终答案里保留关键中间式。",
        ],
    },
    "medicine": {
        "bases": [
            "下面是一个临床问答场景：35 岁女性，发热 39.1 摄氏度、咳嗽 5 天、血氧 95%，既往无基础病。请给出分诊优先级、需要追问的信息和初步鉴别诊断。",
            "请分析一个急诊场景：62 岁男性胸痛 40 分钟，冷汗、恶心，既往高血压。你会如何按优先级组织首轮评估与检查？",
            "给你一段儿科描述：4 岁儿童呕吐腹泻 1 天，精神稍差，尿量减少。请判断脱水程度，指出需要监测的危险信号。",
            "请从基层门诊视角分析：糖尿病患者近 3 个月 HbA1c 明显升高，但自述用药规律。你会怎样排查依从性、饮食、并发感染和监测误差？",
            "一个呼吸科病例：长期吸烟者最近 2 周气促加重，夜间明显，伴少量黄痰。请给出最可能的鉴别方向和首轮检查清单。",
            "请解释抗菌药物使用中的一个决策问题：社区获得性肺炎患者在经验性用药 48 小时后体温下降不明显。你会怎样判断是疗效不足、病原不匹配还是病程尚早？",
        ],
        "modifiers": [
            "请明确标注哪些内容只是一般信息，不能替代医生诊疗。",
            "请按病史、体征、检查、风险分层四段组织回答。",
            "请优先列出需要尽快排除的危险情况。",
            "请把回答写得适合临床实习生阅读。",
            "请把需要追问的内容尽量具体化。",
            "请避免给出不必要的过度检查。",
        ],
    },
    "creative": {
        "bases": [
            "请写一个短篇故事开头：背景是近未来海上城市，主角是一名维护潮汐发电阵列的工程师，今晚他发现城市在悄悄改变自己的地图。",
            "请写一段人物独白：一位退休法官在整理旧卷宗时，突然意识到自己年轻时可能误判过一桩案件。",
            "请设计一段世界观设定：在这个世界里，记忆可以像图书一样借阅，但每借出一次都会轻微磨损原主人的自我认知。",
            "请写一段电影感强的场景描写：凌晨 4 点的高铁维修基地，远处有雷暴，主角必须在 8 分钟内完成一次危险切换。",
            "请模仿偏冷静克制的科幻文风，写一段关于深空探测器失联后重新发回信号的片段。",
            "请为一个悬疑播客写导语：内容围绕一座县城、一个废弃的电台和连续 17 年在同一天播出的同一段求救录音。",
        ],
        "modifiers": [
            "请把篇幅控制在 250 到 350 字。",
            "请避免俗套反转，把重点放在气氛和人物反应。",
            "请加入一个具体可感知的物理细节。",
            "请让语气克制，不要过度煽情。",
            "请在结尾留下一个明确悬念。",
            "请尽量让场景具有电影镜头感。",
        ],
    },
}


DEFAULT_SCENARIOS = ["stable", "block_shift", "medium_shift", "severe_shift"]


def available_domains() -> list[str]:
    return sorted(DOMAIN_BLUEPRINTS)


def make_builtin_prompt_bank(
    domains: list[str] | None = None,
    prompts_per_domain: int = 8,
    seed: int = 7,
) -> list[PromptSpec]:
    domains = domains or available_domains()
    rng = random.Random(seed)
    prompts: list[PromptSpec] = []

    for domain in domains:
        if domain not in DOMAIN_BLUEPRINTS:
            raise ValueError(f"unknown domain: {domain}")
        blueprint = DOMAIN_BLUEPRINTS[domain]
        pairs = [
            (base_idx, modifier_idx)
            for base_idx in range(len(blueprint["bases"]))
            for modifier_idx in range(len(blueprint["modifiers"]))
        ]
        rng.shuffle(pairs)
        for prompt_idx in range(prompts_per_domain):
            base_idx, modifier_idx = pairs[prompt_idx % len(pairs)]
            text = (
                f"{blueprint['bases'][base_idx].strip()}\n\n"
                f"额外要求：{blueprint['modifiers'][modifier_idx].strip()}"
            )
            prompts.append(
                PromptSpec(
                    prompt_id=f"{domain}_{prompt_idx:03d}",
                    domain=domain,
                    text=text,
                )
            )

    return prompts


def load_custom_prompt_bank(dataset_path: str | Path) -> list[PromptSpec]:
    path = Path(dataset_path)
    prompts: list[PromptSpec] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_idx, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            item = json.loads(line)
            domain = item["domain"].strip()
            text = item["text"].strip()
            prompt_id = item.get("id", f"{domain}_{line_idx:03d}")
            prompts.append(PromptSpec(prompt_id=prompt_id, domain=domain, text=text))

    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts


def load_workload_plan(
    prompt_bank: list[PromptSpec],
    plan_path: str | Path,
    seed: int = 7,
) -> dict[str, list[WorkloadItem]]:
    path = Path(plan_path)
    plan = json.loads(path.read_text(encoding="utf-8"))
    prompts_by_domain: dict[str, list[PromptSpec]] = defaultdict(list)
    prompt_lookup = {prompt.prompt_id: prompt for prompt in prompt_bank}
    for prompt in prompt_bank:
        prompts_by_domain[prompt.domain].append(prompt)

    domain_offsets = {domain: 0 for domain in prompts_by_domain}
    workload_map: dict[str, list[WorkloadItem]] = {}

    for scenario_index, (scenario_name, stages) in enumerate(plan.items()):
        rng = random.Random(seed + scenario_index * 104729)
        workload: list[WorkloadItem] = []
        position = 0
        for window_index, stage in enumerate(stages):
            prompt_ids = _expand_stage_to_prompt_ids(
                stage=stage,
                prompts_by_domain=prompts_by_domain,
                prompt_lookup=prompt_lookup,
                domain_offsets=domain_offsets,
                rng=rng,
            )
            for prompt_id in prompt_ids:
                prompt = prompt_lookup[prompt_id]
                workload.append(
                    WorkloadItem(
                        scenario=scenario_name,
                        window_index=window_index,
                        position=position,
                        prompt_id=prompt.prompt_id,
                        domain=prompt.domain,
                    )
                )
                position += 1
        workload_map[scenario_name] = workload

    return workload_map


def build_workloads(
    prompt_bank: list[PromptSpec],
    scenarios: list[str] | None = None,
    num_windows: int = 6,
    window_size: int = 6,
    seed: int = 7,
) -> dict[str, list[WorkloadItem]]:
    if num_windows <= 0 or window_size <= 0:
        raise ValueError("num_windows and window_size must be positive")

    scenarios = scenarios or list(DEFAULT_SCENARIOS)
    prompts_by_domain: dict[str, list[PromptSpec]] = {}
    for prompt in prompt_bank:
        prompts_by_domain.setdefault(prompt.domain, []).append(prompt)

    domains = sorted(prompts_by_domain)
    if len(domains) < 2 and any(scenario != "stable" for scenario in scenarios):
        raise ValueError("at least two domains are required to create workload shifts")

    workload_map: dict[str, list[WorkloadItem]] = {}
    for scenario_idx, scenario in enumerate(scenarios):
        rng = random.Random(seed + scenario_idx * 9973)
        workload: list[WorkloadItem] = []
        position = 0
        anchor_domain = domains[0]

        for window_index in range(num_windows):
            if scenario == "stable":
                sampled_domains = [anchor_domain for _ in range(window_size)]
            else:
                dominant = domains[window_index % len(domains)]
                sampled_domains = _sample_shift_window(
                    domains=domains,
                    dominant=dominant,
                    scenario=scenario,
                    size=window_size,
                    rng=rng,
                )

            for domain in sampled_domains:
                prompt = rng.choice(prompts_by_domain[domain])
                workload.append(
                    WorkloadItem(
                        scenario=scenario,
                        window_index=window_index,
                        position=position,
                        prompt_id=prompt.prompt_id,
                        domain=domain,
                    )
                )
                position += 1

        workload_map[scenario] = workload

    return workload_map


def _expand_stage_to_prompt_ids(
    *,
    stage: dict[str, object],
    prompts_by_domain: dict[str, list[PromptSpec]],
    prompt_lookup: dict[str, PromptSpec],
    domain_offsets: dict[str, int],
    rng: random.Random,
) -> list[str]:
    if "prompt_ids" in stage:
        prompt_ids = [str(prompt_id) for prompt_id in stage["prompt_ids"]]
        for prompt_id in prompt_ids:
            if prompt_id not in prompt_lookup:
                raise ValueError(f"unknown prompt_id in workload plan: {prompt_id}")
        if stage.get("shuffle", False):
            rng.shuffle(prompt_ids)
        return prompt_ids

    if "domain" in stage and "count" in stage:
        return _take_from_domain(
            domain=str(stage["domain"]),
            count=int(stage["count"]),
            prompts_by_domain=prompts_by_domain,
            domain_offsets=domain_offsets,
        )

    if "counts" in stage:
        prompt_ids: list[str] = []
        counts = stage["counts"]
        if not isinstance(counts, dict):
            raise ValueError("workload plan field 'counts' must be a JSON object")
        for domain, count in counts.items():
            prompt_ids.extend(
                _take_from_domain(
                    domain=str(domain),
                    count=int(count),
                    prompts_by_domain=prompts_by_domain,
                    domain_offsets=domain_offsets,
                )
            )
        if stage.get("shuffle", False):
            rng.shuffle(prompt_ids)
        return prompt_ids

    raise ValueError(
        "invalid workload plan stage: expected one of "
        "{prompt_ids}, {domain + count}, or {counts}"
    )


def _take_from_domain(
    *,
    domain: str,
    count: int,
    prompts_by_domain: dict[str, list[PromptSpec]],
    domain_offsets: dict[str, int],
) -> list[str]:
    if domain not in prompts_by_domain:
        raise ValueError(f"unknown domain in workload plan: {domain}")
    if count <= 0:
        raise ValueError("stage count must be positive")

    prompts = prompts_by_domain[domain]
    start = domain_offsets[domain]
    output: list[str] = []
    for offset in range(count):
        prompt = prompts[(start + offset) % len(prompts)]
        output.append(prompt.prompt_id)
    domain_offsets[domain] = (start + count) % len(prompts)
    return output


def _sample_shift_window(
    domains: list[str],
    dominant: str,
    scenario: str,
    size: int,
    rng: random.Random,
) -> list[str]:
    next_domain = domains[(domains.index(dominant) + 1) % len(domains)]
    previous_domain = domains[(domains.index(dominant) - 1) % len(domains)]
    weights = {domain: 0.0 for domain in domains}

    if scenario == "block_shift":
        weights[dominant] = 1.0
    elif scenario == "medium_shift":
        weights[dominant] = 0.65
        weights[next_domain] = 0.25
        noise_domains = [domain for domain in domains if domain not in {dominant, next_domain}]
        shared_noise = 0.10 / len(noise_domains)
        for domain in noise_domains:
            weights[domain] = shared_noise
    elif scenario == "severe_shift":
        weights[dominant] = 0.40
        weights[next_domain] = 0.25
        weights[previous_domain] = 0.20
        noise_domains = [domain for domain in domains if domain not in {dominant, next_domain, previous_domain}]
        if noise_domains:
            shared_noise = 0.15 / len(noise_domains)
            for domain in noise_domains:
                weights[domain] = shared_noise
        else:
            weights[dominant] += 0.15
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    ordered_domains = rng.choices(domains, weights=[weights[domain] for domain in domains], k=size)
    if scenario != "block_shift":
        rng.shuffle(ordered_domains)
    return ordered_domains
