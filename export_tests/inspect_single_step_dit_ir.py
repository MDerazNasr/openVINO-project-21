from collections import Counter
import openvino as ov


IR_PATH = "/Users/mderaznasr/Documents/GitHub/openVINO-project-21/artifacts/openvino_ir/single_step_dit.xml"


def main():
    core = ov.Core()
    model = core.read_model(IR_PATH)

    ops = Counter()
    for op in model.get_ops():
        ops[op.get_type_name()] += 1

    print("[INFO] Operator counts:")
    for name, count in ops.most_common():
        print(f"{name}: {count}")

    print("\n[INFO] First 100 ops:")
    for i, op in enumerate(model.get_ops()[:100]):
        print(f"{i} {op.get_friendly_name()} {op.get_type_name()}")


if __name__ == "__main__":
    main()
