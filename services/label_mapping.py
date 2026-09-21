"""Pemetaan nama kelas model ke label yang ditampilkan aplikasi."""


# Koreksi keluaran aplikasi saja. Nama kelas dan class ID di best.pt tetap.
DISPLAY_LABEL_SWAP = {
    "bakso": "sop ayam",
    "sop ayam": "bakso",
}


def get_display_label(model_class_name: str) -> str:
    normalized_name = " ".join(model_class_name.split()).casefold()
    return DISPLAY_LABEL_SWAP.get(normalized_name, model_class_name)
