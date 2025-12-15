from difflib import SequenceMatcher
from enum import Enum
from typing import List
import re

class UMLClassStereoType (Enum):
    INTERFACE = "<<interface>>"
    ABSTRACT = "<<abstract>>"
    ENUMERATION = "<<enumeration>>"
    EXCEPTION = "<<exception>>"
    ENTITY = "<<entity>>"
    BOUNDARY = "<<boundary>>"
    CONTROL = "<<control>>"
    SERVICE = "<<service>>"
    MICROSERVICE = "<<microservice>>"
    DATABASE = "<<database>>"
    UTILITY = "<<utility>>"
    JAVA_CLASS = "<<Java Class>>"
    ACTOR = "<<actor>>"
    STEREOTYPE = "<<stereotype>>"

    @staticmethod
    def find_closest_match(text: str):
        if text is None:
            return None

        # Use Levenshtein distance
        def similarity(a, b):
            return SequenceMatcher(None, a.lower(), b.lower()).ratio()

        best_match = None
        best_score = 0

        for st in UMLClassStereoType:
            score = similarity(text, st.value)
            if score > best_score:
                best_score = score
                best_match = st

        if best_score > 0.5:
            print(f"Matched stereotype \"{text}\" to \"{best_match}\" with score {best_score}.")
            return best_match
        else:
            print(f"Found no good enough match for stereotype \"{text}\". Best match was \"{best_match}\" with low score {best_score}.")
            return None


class UMLClass:
    def __init__(self, stereotype: UMLClassStereoType, name: str, attributes: List[str] = None, methods: List[str] = None):
        self.stereotype = stereotype
        self.original_name = name
        self.name = self._sanitize_name(name)
        self.attributes = attributes or []
        self.methods = methods or []

    def __repr__(self):
        return f"UMLClass({self.name})"

    @staticmethod
    def _sanitize_name(name: str) -> str:
        # Replace invalid characters with underscores
        sanitized = re.sub(r'[^\w<> ]|^(?=\d)', '_', name)
        return sanitized

class UMLRelationshipType(Enum):
    ASSOCIATION = "--"
    ASSOCIATION_DIRECTIONAL = "-->"
    GENERALIZATION = "--|>"
    AGGREGATION = "--o"
    COMPOSITION = "--*"

    @staticmethod
    def from_end_shape(end_shape: str):
        mapping = {
            "Empty End": UMLRelationshipType.ASSOCIATION,
            "Arrow End": UMLRelationshipType.ASSOCIATION_DIRECTIONAL,
            "Triangle End": UMLRelationshipType.GENERALIZATION,
            "Empty Diamond End": UMLRelationshipType.AGGREGATION,
            "Filled Diamond End": UMLRelationshipType.COMPOSITION,
        }
        return mapping.get(end_shape)


class UMLRelationship:
    def __init__(self,
                 source: UMLClass,
                 target: UMLClass,
                 type_: UMLRelationshipType,
                 role_source: str,
                 multiplicity_source: str,
                 role_target: str,
                 multiplicity_target: str,
                 name: str):
        self.source = source
        self.target = target
        self.type_ = type_

        self.original_role_source = role_source
        self.role_source = self._sanitize(role_source)
        self.multiplicity_source = multiplicity_source

        self.original_role_target = role_target
        self.role_target = self._sanitize(role_target)
        self.multiplicity_target = multiplicity_target

        self.original_name = name
        self.name = self._sanitize(name)

    def __repr__(self):
        return f"{self.source.name} -({self.type_})-> {self.target.name}"

    @staticmethod
    def _sanitize(name: str) -> str:
        sanitized = name.replace("\"", "\"\"")
        return sanitized


class UMLDiagram:
    def __init__(self):
        self.classes: List[UMLClass] = []
        self.relationships: List[UMLRelationship] = []

    def add_class(self, uml_class: UMLClass):
        self.classes.append(uml_class)

    def add_relationship(self, relationship: UMLRelationship):
        self.relationships.append(relationship)

    def get_class(self, index: int) -> UMLClass:
        return self.classes[index]

    def to_plantuml(self) -> str:
        lines = ["@startuml"]

        for uml_class in self.classes:
            header_line = f"class \"{uml_class.name}\""
            if uml_class.stereotype:
                header_line += f" {uml_class.stereotype.value}"
            header_line += " {"
            lines.append(header_line)
            for attribute in uml_class.attributes:
                lines.append(f"\t{attribute}")
            for method in uml_class.methods:
                lines.append(f"\t{method}")
            lines.append("}")

        for rel in self.relationships:
            line = f"\"{rel.source.name}\""

            if rel.role_source or rel.multiplicity_source:
                line += ' "'
                if rel.role_source:
                    line += rel.role_source
                if rel.role_source and rel.multiplicity_source:
                    line += " "
                if rel.multiplicity_source:
                    line += rel.multiplicity_source
                line += '"'

            line += f" {rel.type_.value}"

            if rel.role_target or rel.multiplicity_target:
                line += ' "'
                if rel.role_target:
                    line += rel.role_target
                if rel.role_target and rel.multiplicity_target:
                    line += " "
                if rel.multiplicity_target:
                    line += rel.multiplicity_target
                line += '"'

            line += " \"" + rel.target.name + "\""

            if rel.name:
                line += " : \"" + rel.name + "\""
            lines.append(line)

        lines.append("@enduml")
        return "\n".join(lines)