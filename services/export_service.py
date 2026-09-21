from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path
import csv
import html
import json


@dataclass
class ExportArtifact:
    content: BytesIO
    filename: str
    mimetype: str


class ExportService:
    def __init__(self, database_service) -> None:
        self.database_service = database_service

    @staticmethod
    def _filename(extension: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"food-recognition-{timestamp}.{extension}"

    @staticmethod
    def _flat_scan(record: dict) -> dict:
        timing = record.get("timings") or {}
        nutrition = record.get("nutrition") or {}
        per_100g = nutrition.get("per_100g") or {}
        totals = nutrition.get("totals") or {}
        return {
            "scan_id": record.get("scan_id"),
            "timestamp": record.get("timestamp"),
            "source": record.get("source"),
            "detected": record.get("detected"),
            "food_name": record.get("food_name"),
            "confidence_percent": record.get("confidence_percent"),
            "confidence_ratio": record.get("confidence_ratio"),
            "weight_grams": record.get("weight_grams"),
            "object_count": record.get("object_count"),
            "model_name": record.get("model_name"),
            "model_input": record.get("model_input"),
            "capture_ms": timing.get("capture_ms"),
            "preprocess_ms": timing.get("preprocess_ms"),
            "inference_ms": timing.get("inference_ms"),
            "postprocess_ms": timing.get("postprocess_ms"),
            "weight_ms": timing.get("weight_ms"),
            "complete_ms": timing.get("complete_ms"),
            "nutrition_available": nutrition.get("available"),
            "nutrition_data_version": nutrition.get("data_version"),
            "energy_kcal_per_100g": per_100g.get("energy_kcal"),
            "protein_g_per_100g": per_100g.get("protein_g"),
            "fat_g_per_100g": per_100g.get("fat_g"),
            "carbohydrate_g_per_100g": per_100g.get("carbohydrate_g"),
            "energy_kcal_total": totals.get("energy_kcal"),
            "protein_g_total": totals.get("protein_g"),
            "fat_g_total": totals.get("fat_g"),
            "carbohydrate_g_total": totals.get("carbohydrate_g"),
            "original_image_path": record.get("original_image_path"),
            "annotated_image_path": record.get("annotated_image_path"),
        }

    def csv(self, since=None) -> ExportArtifact:
        dataset = self.database_service.export_dataset(since=since)
        detections_by_scan: dict[str, list[dict]] = {}
        for detection in dataset["detections"]:
            detections_by_scan.setdefault(detection["scan_id"], []).append(
                detection
            )

        records = []
        for scan in dataset["scans"]:
            row = self._flat_scan(scan)
            row["detections_json"] = json.dumps(
                detections_by_scan.get(scan["scan_id"], []),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            records.append(row)

        output = StringIO(newline="")
        fieldnames = list(records[0]) if records else list(
            self._flat_scan({}).keys()
        ) + ["detections_json"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
        content = BytesIO(("\ufeff" + output.getvalue()).encode("utf-8"))
        return ExportArtifact(
            content=content,
            filename=self._filename("csv"),
            mimetype="text/csv; charset=utf-8",
        )

    def xlsx(self, since=None) -> ExportArtifact:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill

        dataset = self.database_service.export_dataset(since=since)
        workbook = Workbook()
        scans_sheet = workbook.active
        scans_sheet.title = "Scans"
        scan_rows = [self._flat_scan(item) for item in dataset["scans"]]
        self._fill_sheet(scans_sheet, scan_rows, Font, PatternFill)
        detections_sheet = workbook.create_sheet("Detections")
        self._fill_sheet(
            detections_sheet,
            dataset["detections"],
            Font,
            PatternFill,
        )

        output = BytesIO()
        workbook.save(output)
        output.seek(0)
        return ExportArtifact(
            content=output,
            filename=self._filename("xlsx"),
            mimetype=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
        )

    @staticmethod
    def _fill_sheet(sheet, records, Font, PatternFill) -> None:
        if not records:
            sheet.append(["Tidak ada data"])
            return
        headers = list(records[0])
        sheet.append(headers)
        for record in records:
            sheet.append([record.get(header) for header in headers])
        fill = PatternFill("solid", fgColor="163A4A")
        for cell in sheet[1]:
            cell.font = Font(color="FFFFFF", bold=True)
            cell.fill = fill
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for column in sheet.columns:
            values = [str(cell.value or "") for cell in column[:100]]
            width = min(45, max(10, max(map(len, values), default=10) + 2))
            sheet.column_dimensions[column[0].column_letter].width = width

    def pdf(self, since=None) -> ExportArtifact:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        dataset = self.database_service.export_dataset(since=since)
        detections_by_scan: dict[str, list[dict]] = {}
        for detection in dataset["detections"]:
            detections_by_scan.setdefault(detection["scan_id"], []).append(
                detection
            )

        output = BytesIO()
        document = SimpleDocTemplate(
            output,
            pagesize=landscape(A4),
            leftMargin=12 * mm,
            rightMargin=12 * mm,
            topMargin=12 * mm,
            bottomMargin=12 * mm,
            title="Laporan Food Recognition",
        )
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#163A4A"),
        )
        story = [
            Paragraph("Laporan Food Recognition", title_style),
            Paragraph(
                f"Dibuat {datetime.now().strftime('%d-%m-%Y %H:%M:%S')} · "
                f"{len(dataset['scans'])} scan",
                styles["Normal"],
            ),
            Spacer(1, 6 * mm),
        ]

        if not dataset["scans"]:
            story.append(Paragraph("Belum ada data scan.", styles["Normal"]))

        for index, scan in enumerate(dataset["scans"]):
            flat = self._flat_scan(scan)
            nutrition_status = (
                "Tersedia" if flat["nutrition_available"] else "Tidak tersedia"
            )
            details = [
                ["Scan ID", flat["scan_id"], "Waktu", flat["timestamp"]],
                ["Makanan", flat["food_name"], "Sumber", flat["source"]],
                [
                    "Confidence",
                    f"{float(flat['confidence_percent'] or 0):.2f}%",
                    "Berat",
                    f"{float(flat['weight_grams'] or 0):.2f} g",
                ],
                [
                    "Objek",
                    flat["object_count"],
                    "Model",
                    f"{flat['model_name'] or '-'} / {flat['model_input'] or '-'}",
                ],
                [
                    "Waktu inferensi",
                    f"{float(flat['inference_ms'] or 0):.2f} ms",
                    "Total proses",
                    f"{float(flat['complete_ms'] or 0):.2f} ms",
                ],
                ["Nutrisi", nutrition_status, "Versi", flat["nutrition_data_version"] or "-"],
                [
                    "Total energi",
                    f"{float(flat['energy_kcal_total'] or 0):.3f} kkal",
                    "Protein / Lemak / Karbo",
                    (
                        f"{float(flat['protein_g_total'] or 0):.3f} / "
                        f"{float(flat['fat_g_total'] or 0):.3f} / "
                        f"{float(flat['carbohydrate_g_total'] or 0):.3f} g"
                    ),
                ],
                [
                    "Foto asli",
                    Paragraph(html.escape(str(flat["original_image_path"] or "-")), styles["Normal"]),
                    "Foto hasil",
                    Paragraph(html.escape(str(flat["annotated_image_path"] or "-")), styles["Normal"]),
                ],
            ]
            table = Table(details, colWidths=[28 * mm, 78 * mm, 34 * mm, 110 * mm])
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E7F2F5")),
                        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#E7F2F5")),
                        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#9CB9C4")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("PADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.extend([table, Spacer(1, 4 * mm)])

            detection_rows = [["Class ID", "Nama", "Confidence", "Bounding box"]]
            for detection in detections_by_scan.get(scan["scan_id"], []):
                detection_rows.append(
                    [
                        detection["class_id"],
                        detection["class_name"],
                        f"{float(detection['confidence_percent']):.2f}%",
                        (
                            f"({detection['x1']}, {detection['y1']}) - "
                            f"({detection['x2']}, {detection['y2']})"
                        ),
                    ]
                )
            if len(detection_rows) == 1:
                detection_rows.append(["-", "Tidak ada deteksi", "-", "-"])
            detection_table = Table(
                detection_rows,
                colWidths=[25 * mm, 80 * mm, 35 * mm, 70 * mm],
            )
            detection_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#163A4A")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#9CB9C4")),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("PADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(detection_table)
            if index < len(dataset["scans"]) - 1:
                story.append(PageBreak())

        document.build(story)
        output.seek(0)
        return ExportArtifact(
            content=output,
            filename=self._filename("pdf"),
            mimetype="application/pdf",
        )

    def create(self, export_format: str, since=None) -> ExportArtifact:
        creators = {
            "csv": self.csv,
            "xlsx": self.xlsx,
            "pdf": self.pdf,
        }
        try:
            creator = creators[export_format.lower()]
        except KeyError as error:
            raise ValueError("Format export harus csv, xlsx, atau pdf.") from error
        return creator(since=since)
