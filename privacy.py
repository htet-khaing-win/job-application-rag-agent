from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer.entities import OperatorConfig

class PIIGuard:
    def __init__(self):
        # tell presidio to use the en_core_web_lg
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        }

        provider = NlpEngineProvider(nlp_configuration = configuration)
        nlp_engine = provider.create_engine()

        self.analyzer = AnalyzerEngine(
            nlp_engine = nlp_engine,
            default_score_threshold = 0.5)
        
        self.anonymizer = AnonymizerEngine()
        self.add_social_recognizer()

        

    def add_social_recognizer(self):
        """ Logic that handel the custom entities for masking"""
        # Defining Patterns
        linkedin_pattern = Pattern(
            name = "LinedIn_pattern",
            regex = r"(?:https?:\/\/)?(?:www\.)?linkedin\.com\/in\/[a-z0-9_-]+\/?",
            score = 0.8
        )

        github_pattern = Pattern(
            name = "GitHub_pattern",
            regex = r"(?:https?:\/\/)?(?:www\.)?github\.com\/in\/[a-z0-9_-]+\/?",
            score = 0.8
        )

        self.analyzer.registry.add_recognizer(PatternRecognizer
                                              (supported_entity="LINKEDIN", 
                                               patterns=[linkedin_pattern]))
        self.analyzer.registry.add_recognizer(PatternRecognizer
                                              (supported_entity="GITHUB", 
                                               patterns=[github_pattern]))


    def redact_text(self, text: str) -> str:

        # Populate target fields to mask
        entities = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION", "LINKEDIN", "GITHUB"]

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
                "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL]"}),
                "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE]"}),
                "LOCATION": OperatorConfig("replace", {"new_value": "[LOCATION]"}),
                "LINKEDIN": OperatorConfig("replace", {"new_value": "[LINKEDIN_URL]"}),
                "GITHUB": OperatorConfig("replace", {"new_value": "[GITHUB_URL]"}),
            }
        ).text