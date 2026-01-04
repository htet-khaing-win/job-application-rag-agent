from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer.entities import OperatorConfig

class PIIGuard:
    def __init__(self):
        # tell presidio to use the en_core_web_trf
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_trf"}],
        }

        provider = NlpEngineProvider(nlp_configuration = configuration)
        nlp_engine = provider.create_engine()

        self.analyzer = AnalyzerEngine(
            nlp_engine = nlp_engine,
            default_score_threshold = 0.5)
        
        self.anonymizer = AnonymizerEngine()
        self.add_social_recognizer()
        self.add_burmese_name_support()

    def add_burmese_name_support(self):
        BURMESE_NAMES = [
            # A-G
            "Aung", "Aye", "Arkar", "Bo", "Ba", "Chan", "Chit", "cho", "Ei", 
            # H-M
            "Htet", "Hla", "Htoo", "Han", "Hone", "Hose", "Khaing", "Kyaw", "Khine", 
            "Khun", "Kaung", "Lin", "Lwin", "Linn", "Min", "Myat", "May", "Maung", 
            # N-S
            "Nanda", "Nay", "Ni", "Nilar", "Nu", "Nan", "Oo", "Okkar", "Phyo", 
            "Paing", "Pyae", "Phoe", "Su", "Sanda", "Sandar", "Sithu", "Soe", 
            # T-Z
            "Thura", "Thet", "Thant", "Tun", "Thiha", "Tint", "Wai", "Win", "Wut", "Yati", 
            "Ye", "Yunn", "Zayar", "Zaw", "Zin", "Zun"
        ]

        name_joined = "|".join(BURMESE_NAMES)
        burmese_regex = rf"\b({name_joined})(?:\s+({name_joined}))+\b"

        burmese_pattern = Pattern(
            name="burmese_name_dictionary_pattern",
            regex=burmese_regex,
            score=0.95  # High score because it's a dictionary match
        )

        self.analyzer.registry.add_recognizer(PatternRecognizer
                                              (supported_entity="BURMESE_NAME", 
                                               patterns=[burmese_pattern]))


    def add_social_recognizer(self):
        """ Logic that handel the custom entities for masking"""
        # Defining Patterns
        linkedin_pattern = Pattern(
            name = "LinedIn_pattern",
            regex = r"linkedin\.com\/in\/[A-Za-z0-9_-]+",
            score = 0.8
        )

        phone_pattern = Pattern(
        name="phone_number_pattern",
        regex= r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\b\d{6,15}\b",
        score=0.8
    )

        github_pattern = Pattern(
            name = "GitHub_pattern",
            regex = r"github\.com\/[A-Za-z0-9_-]+",
            score = 0.8
        )

        cert_pattern = Pattern(
            name = "Cert_pattern",
            regex=r"(?:https?:\/\/)?(?:www\.)?(?:credly\.com|coursera\.org\/verify|badges\.alignment\.org)\/\S+", 
            score=0.8
        )

        self.analyzer.registry.add_recognizer(PatternRecognizer
                                              (supported_entity="LINKEDIN", 
                                               patterns=[linkedin_pattern]))
        
        self.analyzer.registry.add_recognizer(PatternRecognizer
                                              (supported_entity="PHONE_NUMBER", 
                                               patterns=[phone_pattern]))
        
        self.analyzer.registry.add_recognizer(PatternRecognizer
                                              (supported_entity="GITHUB", 
                                               patterns=[github_pattern]))
        
        self.analyzer.registry.add_recognizer(PatternRecognizer
                                              (supported_entity="CERTIFICATE", 
                                               patterns=[cert_pattern]))


    def redact_text(self, text: str) -> str:

        # Populate target fields to mask
        entities = ["PERSON", "BURMESE_NAME", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION", "LINKEDIN", "GITHUB", "CERTIFICATE"]

        # Analyze for PII content
        results = self.analyzer.analyze(
            text = text,
            language = 'en',
            entities = entities
        )

        return self.anonymizer.anonymize(
            text = text,
            analyzer_results = results,
            operators = {
                "PERSON": OperatorConfig("replace", {"new_value": "[CANDIDATE_NAME]"}),
                "BURMESE_NAME": OperatorConfig("replace", {"new_value": "[CANDIDATE_NAME]"}),
                "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL]"}),
                "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE_NUMBER]"}),
                "LOCATION": OperatorConfig("replace", {"new_value": "[LOCATION]"}),
                "LINKEDIN": OperatorConfig("replace", {"new_value": "[LINKEDIN_URL]"}),
                "GITHUB": OperatorConfig("replace", {"new_value": "[GITHUB_URL]"}),
                "CERTIFICATE": OperatorConfig("replace", {"new_value": "[CERTIFICATE_URL]"}),
            }
        ).text