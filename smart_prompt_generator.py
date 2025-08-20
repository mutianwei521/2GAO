#!/usr/bin/env python3
"""
智能prompt生成器
根据物体类别和选择的缺陷类型生成智能prompt
"""

def generate_smart_prompt(category, selected_bad_info):
    """根据物体类别和选择的缺陷类型生成智能prompt"""

    # 预定义的产品词汇（包含imageMVTEC中的所有产品名字）
    product_tokens = {
        "bottle": "bottle",
        "cable": "cable",
        "capsule": "capsule",
        "carpet": "carpet",
        "grid": "grid",
        "hazelnut": "hazelnut",
        "leather": "leather",
        "metal_nut": "metal nut",
        "pill": "pill",
        "screw": "screw",
        "tile": "tile",
        "toothbrush": "toothbrush",
        "transistor": "transistor",
        "wood": "wood",
        "zipper": "zipper"
    }
    
    # 预定义的异常词汇及其变体，包括MVTEC的所有缺陷类型（combined改为damage）
    anomaly_tokens = {
        # bottle (3种缺陷)
        "broken_large": "broken_large",
        "broken_small": "broken_small",
        "contamination": "contamination",

        # cable (8种缺陷)
        "bent_wire": "bent_wire",
        "cable_swap": "cable_swap",
        "combined": "damage",  # 特殊处理：combined -> damage
        "cut_inner_insulation": "cut_inner_insulation",
        "cut_outer_insulation": "cut_outer_insulation",
        "missing_cable": "missing_cable",
        "missing_wire": "missing_wire",
        "poke_insulation": "poke_insulation",

        # capsule (5种缺陷)
        "crack": "crack",
        "faulty_imprint": "faulty_imprint",
        "poke": "poke",
        "scratch": "scratch",
        "squeeze": "squeeze",

        # carpet (5种缺陷)
        "color": "color",
        "cut": "cut",
        "hole": "hole",
        "metal_contamination": "metal_contamination",
        "thread": "thread",

        # grid (5种缺陷)
        "bent": "bent",
        "broken": "broken",
        "glue": "glue",
        "metal_contamination": "metal_contamination",
        "thread": "thread",

        # hazelnut (4种缺陷)
        "crack": "crack",
        "cut": "cut",
        "hole": "hole",
        "print": "print",

        # leather (5种缺陷)
        "color": "color",
        "cut": "cut",
        "fold": "fold",
        "glue": "glue",
        "poke": "poke",

        # metal_nut (4种缺陷)
        "bent": "bent",
        "color": "color",
        "flip": "flip",
        "scratch": "scratch",

        # pill (6种缺陷)
        "color": "color",
        "contamination": "contamination",
        "crack": "crack",
        "faulty_imprint": "faulty_imprint",
        "pill_type": "pill_type",
        "scratch": "scratch",

        # screw (5种缺陷)
        "manipulated_front": "manipulated_front",
        "scratch_head": "scratch_head",
        "scratch_neck": "scratch_neck",
        "thread_side": "thread_side",
        "thread_top": "thread_top",

        # tile (5种缺陷)
        "crack": "crack",
        "glue_strip": "glue_strip",
        "gray_stroke": "gray_stroke",
        "oil": "oil",
        "rough": "rough",

        # toothbrush (1种缺陷)
        "defective": "defective",

        # transistor (4种缺陷)
        "bent_lead": "bent_lead",
        "cut_lead": "cut_lead",
        "damaged_case": "damaged_case",
        "misplaced": "misplaced",

        # wood (5种缺陷)
        "color": "color",
        "combined": "damage",  # 特殊处理：combined -> damage
        "hole": "hole",
        "liquid": "liquid",
        "scratch": "scratch",

        # zipper (5种缺陷)
        "broken_teeth": "broken_teeth",
        "fabric_border": "fabric_border",
        "fabric_interior": "fabric_interior",
        "rough": "rough",
        "split_teeth": "split_teeth",
        "squeezed_teeth": "squeezed_teeth"
    }
    
    # 获取产品token
    product_token = product_tokens.get(category, category)

    # 提取所有缺陷类型并映射为anomaly tokens
    anomaly_token_list = []
    defect_types = []

    for bad_info in selected_bad_info:
        subfolder = bad_info['subfolder']
        # 获取对应的anomaly token
        anomaly_token = anomaly_tokens.get(subfolder, subfolder)

        if anomaly_token not in anomaly_token_list:
            anomaly_token_list.append(anomaly_token)

        if subfolder not in defect_types:
            defect_types.append(subfolder)

    # 生成最终prompt：产品名 + 所有缺陷类型
    # 格式：[product_token] [anomaly_token1] [anomaly_token2] ...
    all_anomaly_tokens = " ".join(anomaly_token_list)
    prompt = f"{product_token} {all_anomaly_tokens}"

    return prompt, defect_types, anomaly_token_list

def generate_individual_prompts(category, selected_bad_info):
    """为每个缺陷图生成单独的prompt"""

    # 预定义的产品词汇
    product_tokens = {
        "bottle": "bottle",
        "cable": "cable",
        "capsule": "capsule",
        "carpet": "carpet",
        "grid": "grid",
        "hazelnut": "hazelnut",
        "leather": "leather",
        "metal_nut": "metal nut",
        "pill": "pill",
        "screw": "screw",
        "tile": "tile",
        "toothbrush": "toothbrush",
        "transistor": "transistor",
        "wood": "wood",
        "zipper": "zipper"
    }

    # 预定义的异常词汇及其变体，包括MVTEC的所有缺陷类型（combined改为damage）
    anomaly_tokens = {
        # bottle (3种缺陷)
        "broken_large": "broken_large",
        "broken_small": "broken_small",
        "contamination": "contamination",

        # cable (8种缺陷)
        "bent_wire": "bent_wire",
        "cable_swap": "cable_swap",
        "combined": "damage",  # 特殊处理：combined -> damage
        "cut_inner_insulation": "cut_inner_insulation",
        "cut_outer_insulation": "cut_outer_insulation",
        "missing_cable": "missing_cable",
        "missing_wire": "missing_wire",
        "poke_insulation": "poke_insulation",

        # capsule (5种缺陷)
        "crack": "crack",
        "faulty_imprint": "faulty_imprint",
        "poke": "poke",
        "scratch": "scratch",
        "squeeze": "squeeze",

        # carpet (5种缺陷)
        "color": "color",
        "cut": "cut",
        "hole": "hole",
        "metal_contamination": "metal_contamination",
        "thread": "thread",

        # grid (5种缺陷)
        "bent": "bent",
        "broken": "broken",
        "glue": "glue",
        "metal_contamination": "metal_contamination",
        "thread": "thread",

        # hazelnut (4种缺陷)
        "crack": "crack",
        "cut": "cut",
        "hole": "hole",
        "print": "print",

        # leather (5种缺陷)
        "color": "color",
        "cut": "cut",
        "fold": "fold",
        "glue": "glue",
        "poke": "poke",

        # metal_nut (4种缺陷)
        "bent": "bent",
        "color": "color",
        "flip": "flip",
        "scratch": "scratch",

        # pill (6种缺陷)
        "color": "color",
        "contamination": "contamination",
        "crack": "crack",
        "faulty_imprint": "faulty_imprint",
        "pill_type": "pill_type",
        "scratch": "scratch",

        # screw (5种缺陷)
        "manipulated_front": "manipulated_front",
        "scratch_head": "scratch_head",
        "scratch_neck": "scratch_neck",
        "thread_side": "thread_side",
        "thread_top": "thread_top",

        # tile (5种缺陷)
        "crack": "crack",
        "glue_strip": "glue_strip",
        "gray_stroke": "gray_stroke",
        "oil": "oil",
        "rough": "rough",

        # toothbrush (1种缺陷)
        "defective": "defective",

        # transistor (4种缺陷)
        "bent_lead": "bent_lead",
        "cut_lead": "cut_lead",
        "damaged_case": "damaged_case",
        "misplaced": "misplaced",

        # wood (5种缺陷)
        "color": "color",
        "combined": "damage",  # 特殊处理：combined -> damage
        "hole": "hole",
        "liquid": "liquid",
        "scratch": "scratch",

        # zipper (5种缺陷)
        "broken_teeth": "broken_teeth",
        "fabric_border": "fabric_border",
        "fabric_interior": "fabric_interior",
        "rough": "rough",
        "split_teeth": "split_teeth",
        "squeezed_teeth": "squeezed_teeth"
    }

    # 获取产品token
    product_token = product_tokens.get(category, category)

    # 为每个缺陷图生成单独的prompt
    individual_prompts = []

    for bad_info in selected_bad_info:
        subfolder = bad_info['subfolder']
        filename = bad_info['filename']

        # 获取对应的anomaly token
        anomaly_token = anomaly_tokens.get(subfolder, subfolder)

        # 生成单独的prompt
        individual_prompt = f"{product_token} {anomaly_token}"

        individual_prompts.append({
            'subfolder': subfolder,
            'filename': filename,
            'prompt': individual_prompt,
            'anomaly_token': anomaly_token
        })

    return individual_prompts

def test_smart_prompt_generation():
    """测试智能prompt生成功能"""

    print("🧠 Testing Smart Prompt Generation (New Version)")
    print("=" * 60)

    # 测试用例
    test_cases = [
        {
            "category": "bottle",
            "selected_bad_info": [
                {"subfolder": "broken_large", "filename": "001"}
            ],
            "expected_combined": "bottle broken_large",
            "expected_individual": ["bottle broken_large"]
        },
        {
            "category": "cable",
            "selected_bad_info": [
                {"subfolder": "bent_wire", "filename": "001"},
                {"subfolder": "cut_outer_insulation", "filename": "002"}
            ],
            "expected_combined": "cable bent_wire cut_outer_insulation",
            "expected_individual": ["cable bent_wire", "cable cut_outer_insulation"]
        },
        {
            "category": "cable",
            "selected_bad_info": [
                {"subfolder": "combined", "filename": "001"}
            ],
            "expected_combined": "cable damage",  # combined -> damage
            "expected_individual": ["cable damage"]
        },
        {
            "category": "hazelnut",
            "selected_bad_info": [
                {"subfolder": "crack", "filename": "001"}
            ],
            "expected_combined": "hazelnut crack",
            "expected_individual": ["hazelnut crack"]
        },
        {
            "category": "pill",
            "selected_bad_info": [
                {"subfolder": "color", "filename": "001"},
                {"subfolder": "crack", "filename": "002"},
                {"subfolder": "scratch", "filename": "003"}
            ],
            "expected_combined": "pill color crack scratch",
            "expected_individual": ["pill color", "pill crack", "pill scratch"]
        }
    ]
    
    success_count = 0

    for i, test_case in enumerate(test_cases):
        print(f"\n[PARSE] Test Case {i+1}: {test_case['category']}")
        print(f"   Defect info: {test_case['selected_bad_info']}")

        # 测试组合prompt生成
        combined_prompt, defect_types, anomaly_tokens = generate_smart_prompt(
            test_case['category'],
            test_case['selected_bad_info']
        )

        print(f"   [TEST] Combined prompt: '{combined_prompt}'")
        print(f"   [TEST] Expected: '{test_case['expected_combined']}'")

        # 测试单独prompt生成
        individual_prompts = generate_individual_prompts(
            test_case['category'],
            test_case['selected_bad_info']
        )

        print(f"   [TEST] Individual prompts:")
        for j, prompt_info in enumerate(individual_prompts):
            print(f"      {j+1}. {prompt_info['subfolder']}: '{prompt_info['prompt']}'")

        # 验证组合prompt
        combined_success = (combined_prompt == test_case['expected_combined'])
        print(f"   [RESULT] Combined prompt: {'PASS' if combined_success else 'FAIL'}")

        # 验证单独prompts
        individual_success = True
        expected_individual = test_case['expected_individual']
        if len(individual_prompts) == len(expected_individual):
            for j, prompt_info in enumerate(individual_prompts):
                if prompt_info['prompt'] != expected_individual[j]:
                    individual_success = False
                    break
        else:
            individual_success = False

        print(f"   [RESULT] Individual prompts: {'PASS' if individual_success else 'FAIL'}")

        if combined_success and individual_success:
            success_count += 1
    
    print(f"\n[SUMMARY] Results: {success_count}/{len(test_cases)} test cases passed")

    if success_count == len(test_cases):
        print("[SUCCESS] All smart prompt generation tests passed!")
        return True
    else:
        print("[ERROR] Some smart prompt generation tests failed!")
        return False

if __name__ == "__main__":
    test_smart_prompt_generation()
