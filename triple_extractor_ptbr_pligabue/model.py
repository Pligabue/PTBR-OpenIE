import tensorflow as tf
from pathlib import Path
import uuid

from typing import Union

from .constants import DEFAULT_MODEL_NAME, DEFAULT_PRED_THRESHOLD, DEFAULT_ARG_THREHSOLD, DEFAULT_SENTENCE_SIZE
from .argument_prediction import ArgumentPredictor
from .predicate_extraction import PredicateExtractor
from .data_formatter import DataFormatter


class TripleExtractor(DataFormatter):
    def __init__(self, pe_layers_or_name: Union[tuple[int], str], ap_layers_or_name: Union[tuple[int], str], sentence_size: int = DEFAULT_SENTENCE_SIZE) -> None:
        if isinstance(pe_layers_or_name, str):
            self.predicate_extractor = PredicateExtractor.load(pe_layers_or_name)
        else:
            self.predicate_extractor = PredicateExtractor(*pe_layers_or_name, sentence_size=sentence_size)

        if isinstance(ap_layers_or_name, str):
            self.argument_predictor = ArgumentPredictor.load(ap_layers_or_name)
        else:
            self.argument_predictor = ArgumentPredictor(*ap_layers_or_name, sentence_size=sentence_size)

    @classmethod
    def load(cls, name: str = DEFAULT_MODEL_NAME):
        return cls(name, name)

    def save(self, name: str = DEFAULT_MODEL_NAME):
        self.predicate_extractor.save(name)
        self.argument_predictor.save(name)

    def compile(self, pe_optimizer=None, pe_loss=None, pe_metrics=None, ap_optimizer=None, ap_loss=None,
                ap_metrics=None):
        default_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        default_loss = tf.keras.losses.CategoricalCrossentropy()
        default_metrics = [tf.keras.metrics.CategoricalCrossentropy()]

        pe_optimizer = pe_optimizer or ap_optimizer or default_optimizer
        pe_loss = pe_loss or ap_loss or default_loss
        pe_metrics = pe_metrics or ap_metrics or default_metrics

        ap_optimizer = ap_optimizer or pe_optimizer or default_optimizer
        ap_loss = ap_loss or pe_loss or default_loss
        ap_metrics = ap_metrics or pe_metrics or default_metrics

        self.predicate_extractor.compile(optimizer=pe_optimizer, loss=pe_loss, metrics=pe_metrics)
        self.argument_predictor.compile(optimizer=ap_optimizer, loss=ap_loss, metrics=ap_metrics)

    def summary(self):
        self.predicate_extractor.summary()
        self.argument_predictor.summary()

    def fit(self, training_sentences, *args, merge_repeated=False, epochs=20, pe_epochs=None, ap_epochs=None,
            early_stopping=False, callbacks=None, **kwargs):
        pe_epochs = pe_epochs or epochs
        ap_epochs = ap_epochs or epochs

        print("Predicate Extractor:")
        self.predicate_extractor.fit(training_sentences, *args, merge_repeated=merge_repeated, epochs=pe_epochs,
                                     early_stopping=early_stopping, callbacks=callbacks, **kwargs)
        print("\nArgument predictor:")
        self.argument_predictor.fit(training_sentences, *args, merge_repeated=merge_repeated, epochs=ap_epochs,
                                    early_stopping=early_stopping, callbacks=callbacks, **kwargs)

    def predict(self, sentences: list[str], pred_threshold=DEFAULT_PRED_THRESHOLD,
                arg_threshold=DEFAULT_ARG_THREHSOLD):
        arg_pred_inputs = self.predicate_extractor(sentences, acceptance_threshold=pred_threshold)
        outputs = self.argument_predictor(arg_pred_inputs, acceptance_threshold=arg_threshold)
        return outputs

    def annotate_sentences(self, sentences: list[str], pred_threshold=DEFAULT_PRED_THRESHOLD,
                           arg_threshold=DEFAULT_ARG_THREHSOLD):
        outputs = self.predict(sentences, pred_threshold=pred_threshold, arg_threshold=arg_threshold)
        for sentence_id, tokens, pred_masks, subj_mask, obj_mask in outputs:
            annotation = self.build_annotation(sentence_id, tokens, pred_masks, subj_mask, obj_mask)
            print(annotation)
            print()

    def gen_csv(self, sentences: list[str], filepath: Path, pred_threshold=DEFAULT_PRED_THRESHOLD,
                arg_threshold=DEFAULT_ARG_THREHSOLD, title='', id_prefix=''):
        outputs = self.predict(sentences, pred_threshold=pred_threshold, arg_threshold=arg_threshold)
        df = self.build_df(outputs, id_prefix=id_prefix)
        with filepath.open('w', encoding="utf-8") as f:
            f.write(f"# {title}\n")
            df.to_csv(f, sep=";", index=False, lineterminator='\n')
        return df

    def process_doc(self, doc_path: Path, csv_path: Union[Path, None] = None,
                    pred_threshold=DEFAULT_PRED_THRESHOLD, arg_threshold=DEFAULT_ARG_THREHSOLD):
        csv_path = csv_path or doc_path.with_suffix('.csv')

        with doc_path.open(encoding="utf-8") as f:
            doc = f.read()
        sentences = self.doc_to_sentences(doc)
        id_prefix = str(uuid.uuid4())

        return self.gen_csv(
            sentences,
            csv_path,
            title=doc_path.as_posix(),
            id_prefix=id_prefix,
            pred_threshold=pred_threshold,
            arg_threshold=arg_threshold
        )

    def process_docs(self, doc_dir: list[Path], csv_dir: Path,
                     pred_threshold=DEFAULT_PRED_THRESHOLD, arg_threshold=DEFAULT_ARG_THREHSOLD):
        for doc_path in doc_dir:
            self.process_doc(
                doc_path,
                (csv_dir / doc_path.stem).with_suffix(".csv"),
                pred_threshold=pred_threshold,
                arg_threshold=arg_threshold
            )
