#!/usr/bin/env python3
"""
Smart Prompt Generator
Generates intelligent prompts based on object category and selected defect types
"""

# ==============================================================================
# [Prompt Engineering Module]
# Manages domain-specific vocabulary (product/defect tokens) and constructs
# semantically precise prompts for the diffusion model
# ==============================================================================
def generate_smart_prompt(category, selected_bad_info):
    """Generate intelligent prompt based on object category and selected defect types"""

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
    
    # Predefined anomaly vocabulary and variants, including all MVTec defect types (combined -> damage)
    anomaly_tokens = {
        # bottle (3 defect types)
        "broken_large": "broken_large",
        "broken_small": "broken_small",
        "contamination": "contamination",

        # cable (8 defect types)
        "bent_wire": "bent_wire",
        "cable_swap": "cable_swap",
        "combined": "damage",  # Special handling: combined -> damage
        "cut_inner_insulation": "cut_inner_insulation",
        "cut_outer_insulation": "cut_outer_insulation",
        "missing_cable": "missing_cable",
        "missing_wire": "missing_wire",
        "poke_insulation": "poke_insulation",

        # capsule (5 defect types)
        "crack": "crack",
        "faulty_imprint": "faulty_imprint",
        "poke": "poke",
        "scratch": "scratch",
        "squeeze": "squeeze",

        # carpet (5 defect types)
        "color": "color",
        "cut": "cut",
        "hole": "hole",
        "metal_contamination": "metal_contamination",
        "thread": "thread",

        # grid (5 defect types)
        "bent": "bent",
        "broken": "broken",
        "glue": "glue",
        "metal_contamination": "metal_contamination",
        "thread": "thread",

        # hazelnut (4 defect types)
        "crack": "crack",
        "cut": "cut",
        "hole": "hole",
        "print": "print",

        # leather (5 defect types)
        "color": "color",
        "cut": "cut",
        "fold": "fold",
        "glue": "glue",
        "poke": "poke",

        # metal_nut (4 defect types)
        "bent": "bent",
        "color": "color",
        "flip": "flip",
        "scratch": "scratch",

        # pill (6 defect types)
        "color": "color",
        "contamination": "contamination",
        "crack": "crack",
        "faulty_imprint": "faulty_imprint",
        "pill_type": "pill_type",
        "scratch": "scratch",

        # screw (5 defect types)
        "manipulated_front": "manipulated_front",
        "scratch_head": "scratch_head",
        "scratch_neck": "scratch_neck",
        "thread_side": "thread_side",
        "thread_top": "thread_top",

        # tile (5 defect types)
        "crack": "crack",
        "glue_strip": "glue_strip",
        "gray_stroke": "gray_stroke",
        "oil": "oil",
        "rough": "rough",

        # toothbrush (1 defect type)
        "defective": "defective",

        # transistor (4 defect types)
        "bent_lead": "bent_lead",
        "cut_lead": "cut_lead",
        "damaged_case": "damaged_case",
        "misplaced": "misplaced",

        # wood (5 defect types)
        "color": "color",
        "combined": "damage",  # Special handling: combined -> damage
        "hole": "hole",
        "liquid": "liquid",
        "scratch": "scratch",

        # zipper (5 defect types)
        "broken_teeth": "broken_teeth",
        "fabric_border": "fabric_border",
        "fabric_interior": "fabric_interior",
        "rough": "rough",
        "split_teeth": "split_teeth",
        "squeezed_teeth": "squeezed_teeth"
    }
    
    # Get product token
    product_token = product_tokens.get(category, category)

    # Extract all defect types and map to anomaly tokens
    anomaly_token_list = []
    defect_types = []

    for bad_info in selected_bad_info:
        subfolder = bad_info['subfolder']
        # Get corresponding anomaly token
        anomaly_token = anomaly_tokens.get(subfolder, subfolder)

        if anomaly_token not in anomaly_token_list:
            anomaly_token_list.append(anomaly_token)

        if subfolder not in defect_types:
            defect_types.append(subfolder)

    # Generate final prompt: product name + all defect types
    # Format: [product_token] [anomaly_token1] [anomaly_token2] ...
    all_anomaly_tokens = " ".join(anomaly_token_list)
    prompt = f"{product_token} {all_anomaly_tokens}"

    return prompt, defect_types, anomaly_token_list

def generate_individual_prompts(category, selected_bad_info):
    """Generate individual prompts for each defect image"""

    # Predefined product vocabulary
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

    # Predefined anomaly vocabulary and variants, including all MVTec defect types (combined -> damage)
    anomaly_tokens = {
        # bottle (3 defect types)
        "broken_large": "broken_large",
        "broken_small": "broken_small",
        "contamination": "contamination",

        # cable (8 defect types)
        "bent_wire": "bent_wire",
        "cable_swap": "cable_swap",
        "combined": "damage",  # Special handling: combined -> damage
        "cut_inner_insulation": "cut_inner_insulation",
        "cut_outer_insulation": "cut_outer_insulation",
        "missing_cable": "missing_cable",
        "missing_wire": "missing_wire",
        "poke_insulation": "poke_insulation",

        # capsule (5 defect types)
        "crack": "crack",
        "faulty_imprint": "faulty_imprint",
        "poke": "poke",
        "scratch": "scratch",
        "squeeze": "squeeze",

        # carpet (5 defect types)
        "color": "color",
        "cut": "cut",
        "hole": "hole",
        "metal_contamination": "metal_contamination",
        "thread": "thread",

        # grid (5 defect types)
        "bent": "bent",
        "broken": "broken",
        "glue": "glue",
        "metal_contamination": "metal_contamination",
        "thread": "thread",

        # hazelnut (4 defect types)
        "crack": "crack",
        "cut": "cut",
        "hole": "hole",
        "print": "print",

        # leather (5 defect types)
        "color": "color",
        "cut": "cut",
        "fold": "fold",
        "glue": "glue",
        "poke": "poke",

        # metal_nut (4 defect types)
        "bent": "bent",
        "color": "color",
        "flip": "flip",
        "scratch": "scratch",

        # pill (6 defect types)
        "color": "color",
        "contamination": "contamination",
        "crack": "crack",
        "faulty_imprint": "faulty_imprint",
        "pill_type": "pill_type",
        "scratch": "scratch",

        # screw (5 defect types)
        "manipulated_front": "manipulated_front",
        "scratch_head": "scratch_head",
        "scratch_neck": "scratch_neck",
        "thread_side": "thread_side",
        "thread_top": "thread_top",

        # tile (5 defect types)
        "crack": "crack",
        "glue_strip": "glue_strip",
        "gray_stroke": "gray_stroke",
        "oil": "oil",
        "rough": "rough",

        # toothbrush (1 defect type)
        "defective": "defective",

        # transistor (4 defect types)
        "bent_lead": "bent_lead",
        "cut_lead": "cut_lead",
        "damaged_case": "damaged_case",
        "misplaced": "misplaced",

        # wood (5 defect types)
        "color": "color",
        "combined": "damage",  # Special handling: combined -> damage
        "hole": "hole",
        "liquid": "liquid",
        "scratch": "scratch",

        # zipper (5 defect types)
        "broken_teeth": "broken_teeth",
        "fabric_border": "fabric_border",
        "fabric_interior": "fabric_interior",
        "rough": "rough",
        "split_teeth": "split_teeth",
        "squeezed_teeth": "squeezed_teeth"
    }

    # Get product token
    product_token = product_tokens.get(category, category)

    # Generate individual prompt for each defect image
    individual_prompts = []

    for bad_info in selected_bad_info:
        subfolder = bad_info['subfolder']
        filename = bad_info['filename']

        # Get corresponding anomaly token
        anomaly_token = anomaly_tokens.get(subfolder, subfolder)

        # Generate individual prompt
        individual_prompt = f"{product_token} {anomaly_token}"

        individual_prompts.append({
            'subfolder': subfolder,
            'filename': filename,
            'prompt': individual_prompt,
            'anomaly_token': anomaly_token
        })

    return individual_prompts

def test_smart_prompt_generation():
    """Test smart prompt generation functionality"""

    print("🧠 Testing Smart Prompt Generation (New Version)")
    print("=" * 60)

    # Test cases
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

        # Test combined prompt generation
        combined_prompt, defect_types, anomaly_tokens = generate_smart_prompt(
            test_case['category'],
            test_case['selected_bad_info']
        )

        print(f"   [TEST] Combined prompt: '{combined_prompt}'")
        print(f"   [TEST] Expected: '{test_case['expected_combined']}'")

        # Test individual prompt generation
        individual_prompts = generate_individual_prompts(
            test_case['category'],
            test_case['selected_bad_info']
        )

        print(f"   [TEST] Individual prompts:")
        for j, prompt_info in enumerate(individual_prompts):
            print(f"      {j+1}. {prompt_info['subfolder']}: '{prompt_info['prompt']}'")

        # Verify combined prompt
        combined_success = (combined_prompt == test_case['expected_combined'])
        print(f"   [RESULT] Combined prompt: {'PASS' if combined_success else 'FAIL'}")

        # Verify individual prompts
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
