import csv
import io


class unit:
    def __init__(self, name: str, unit_type: int) -> None:
        self.name_ = name
        self.unit_type_ = unit_type

    def get_name(self) -> str:
        return self.name_


class unit_builder:
    def get_units(self) -> list[unit]:
        raise NotImplemented


class file_unit_builder(unit_builder):
    del_ = ";"
    quote_ = "\n"
    name_str = "Name"
    type_str = "Type"

    def __init__(self, file_str: str) -> None:
        self.file_str = file_str

    def from_file_str(self, file_str: str) -> list[unit]:
        with open(file_str, "r") as f:
            unit_list = self.from_file(f)
        return unit_list

    def from_file(self, file: io.TextIOWrapper) -> list[unit]:
        csv_reader = csv.DictReader(
            file, delimiter=file_unit_builder.del_, quotechar=file_unit_builder.quote_
        )
        unit_list = []
        for line in csv_reader:
            current_unit = unit(
                line[file_unit_builder.name_str], int(line[file_unit_builder.type_str])
            )
            unit_list.append(current_unit)
        return unit_list

    def get_units(self) -> list[unit]:
        return self.from_file_str(self.file_str)
