"""Close only the twelve evidenced numerical tasks, preserving physical limits."""

import json
from collections import Counter
from pathlib import Path

from atomic_model_controlled_plant.session import identity
from scripts.extended_plant_release import envelope

TASKS = {"AI-05.03", "AI-05.05", "AI-05.08", "AI-06.06", *[f"AI-07.{n:02d}" for n in range(1, 9)]}


def main():
    summary = envelope("validation/ai050607_controlled_summary_v1.json")
    checks = json.loads(Path("validation/ai050607_controlled_checks_v1.json").read_text())
    installed = json.loads(Path("validation/ai050607_controlled_installed_v1.json").read_text())
    if (
        not summary["all_three_blocks_numerical_acceptance"]
        or installed["source"] != identity()
        or checks["source"] != identity()
        or checks["regression"]["exit_code"] != 0
        or checks["installed_tests"]["exit_code"] != 0
    ):
        raise ValueError("Numerical delivery gates are not complete/current")
    path = Path("validation/ai_industrial_workplan_v1.json")
    plan = json.loads(path.read_text())
    if plan["version"] != "1.75.0":
        raise ValueError("Expected prior workplan1.75.0")
    evidence = [
        "docs/AI_05_06_07_EXTENDED_MODEL_RU.md",
        "validation/ai050607_controlled_summary_v1.json",
        "validation/ai050607_controlled_checks_v1.json",
        "validation/ai050607_controlled_installed_v1.json",
        "validation/ai050607_controlled_delivery_v1.json",
        "validation/ai07_controlled_configuration_v1.json",
        "validation/ai06_pool_b274_independent_import_v2.json",
    ]
    for package in plan["packages"]:
        for task in package["tasks"]:
            if task["id"] in TASKS:
                task["status"] = "accepted"
                task.setdefault("evidence", []).extend(
                    p for p in evidence if p not in task.get("evidence", [])
                )
                task["verification_level"] = "numerical_verification"
                task["acceptance_scope"] = (
                    "Exact source-bound reduced flow-controlled PWR and declared component/HEM-vessel programs; not experimental or real-plant qualification"
                )
                if task["id"] == "AI-06.06":
                    task["experimental_validation_status"] = (
                        "not_established; qualified independent numerical four-phenomenon programs; actual pool B274 not admitted for current forced boiling"
                    )
                plan["acceptance_decisions"].append(
                    dict(
                        date="2026-11-06",
                        task=task["id"],
                        decision="accepted",
                        scope=task["acceptance_scope"],
                        basis=evidence,
                        exclusions=[
                            "Experimental/plant-reference acceptance",
                            "Arbitrary unqualified input programs",
                            "Four-quadrant real-pump/CHF/real-plant-control attestation",
                        ],
                    )
                )
        if package["id"] in ["AI-05", "AI-06", "AI-07"]:
            if not all(t["status"] == "accepted" for t in package["tasks"]):
                raise ValueError("Package still has open tasks")
            package["status"] = "accepted"
            package["acceptance_scope"] = (
                "Declared numerical implementation only; validated=false, plant_referenced=false"
            )
            package["experimental_validation_status"] = "not_established"
            package.setdefault("evidence", []).extend(
                p for p in evidence if p not in package.get("evidence", [])
            )
    plan["version"] = "1.76.0"
    plan["date"] = "2026-11-06"
    plan["validated"] = False
    plan["plant_referenced"] = False
    plan["current_extension_evidence"] = "validation/ai050607_controlled_delivery_v1.json"
    plan["execution_history"].append(
        dict(
            date="2026-11-06",
            version="1.76.0",
            packages=["AI-05", "AI-06", "AI-07"],
            accepted_tasks=sorted(TASKS),
            note="Closed twelve tasks within declared numerical scopes. Ten full current-source programs and complete replay; load7200 rather than falsely accepting all-state equilibrium at600. Experimental validation remains unestablished; old failures retained.",
            evidence=evidence,
        )
    )
    plan["scope_revisions"].append(
        dict(
            date="2026-11-06",
            packages=["AI-05", "AI-06", "AI-07"],
            scope="Numerical-verification fallback already explicit in AI-06.06; package task closure does not promote experimental or actual-plant status",
            physical_validation_status="open; not established",
        )
    )
    path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    counts = Counter(t["status"] for b in plan["packages"] for t in b["tasks"])
    opened = [b for b in plan["packages"] if any(t["status"] != "accepted" for t in b["tasks"])]
    lines = [
        "# Текущий статус и оставшиеся блоки Atomic Sim",
        "",
        "Обновлено: **6 ноября 2026**. Машинный план **1.76.0**. Открытый моделер для ИИ/LLM.",
        "",
        f"**Принято {counts['accepted']} из 109 задач. Осталось {109 - counts['accepted']}: {counts['in_progress']} в работе, {counts['planned']} запланированы.** Открыто блоков: **{len(opened)}**. Количество задач не является процентом промышленной готовности.",
        "",
        "## Закрытая численная поставка AI-05/06/07",
        "",
        "Сеть/оборудование, четыре независимые фазовые задачи и связанная конфигурация `pwr-reduced-flow-controlled` приняты **в объявленной численной области**. Номинал 3600 с двумя методами, выбег 1800 с двумя точностями, SCRAM 60 с, установление после сброса 7200 с, питание/сток 120 с, нулевой напор/естественная циркуляция 1800 с и возмущение теплоотвода 120 с. Полные истории воспроизведены; операторские программы допускаются только по точному профилю свидетельств.",
        "",
        "**Экспериментальная валидация и соответствие реальной станции не заявлены.** AI-06.06 закрыт как numerical verification по предусмотренному исключению; реальные pool-boiling данные проверены, но неприменимы как валидация текущего проточного Chen. Короткий сброс600 и старые отказы ВВЭР не превращены в успех.",
        "",
        "[Описание и ограничения](AI_05_06_07_EXTENDED_MODEL_RU.md), [машинная приёмка](../validation/ai050607_controlled_summary_v1.json), [машинный план](../validation/ai_industrial_workplan_v1.json).",
        "",
        "## Открытые блоки",
        "",
    ]
    for package in opened:
        tasks = [t for t in package["tasks"] if t["status"] != "accepted"]
        lines.extend(
            [
                f"### {package['id']} — {package['title']}",
                "",
                f"Осталось задач: **{len(tasks)}**.",
                "",
            ]
        )
        for task in tasks:
            lines.append(
                f"- [ ] **{task['id']}** — {task['description']} Статус: {task['status']}."
            )
        lines.append("")
    lines.extend(
        [
            "## Следующая работа",
            "",
            "Продолжать пространственную/тепловую связь, независимые экспериментальные данные и UQ, агентские контракты и общую приёмку по оставшимся задачам. Наличие численного эталона не заменяет физическую валидацию.",
            "",
        ]
    )
    Path("docs/CURRENT_STATUS_RU.md").write_text("\n".join(lines))
    update = """Обновление 1.76.0 (6 ноября 2026): [численная поставка AI-05/06/07](AI_05_06_07_EXTENDED_MODEL_RU.md). Приняты двенадцать задач в явно ограниченной исследовательской конфигурации `pwr-reduced-flow-controlled`: прямой ранговый стационар/переключения/rollback, четыре независимые фазовые задачи, десять полных программ с replay. Номинал3600 с, выбег/естественная циркуляция1800 с, SCRAM60 с, питание/сток120 с; сброс доведён до строгого установления на7200 с, прежние600 с не объявлены установившимися. **82/109** задач принято, осталось27 в шести блоках. Экспериментальная валидация и реальная станция остаются неподтверждёнными.\n\n"""
    for name in ["AI_INDUSTRIAL_EXECUTION_PLAN_RU.md", "READINESS_REVIEW_RU.md"]:
        p = Path("docs") / name
        text = p.read_text()
        at = text.find("\n\n") + 2
        p.write_text(text[:at] + update + text[at:])
    print(dict(counts=counts, open_blocks=len(opened)))


if __name__ == "__main__":
    main()
