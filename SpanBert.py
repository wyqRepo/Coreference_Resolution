import torch
from allennlp.data import Vocabulary, allennlp_collate
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.token_indexers import PretrainedTransformerMismatchedIndexer
from allennlp.models import Model
from allennlp.modules import FeedForward
from allennlp.modules.seq2seq_encoders import PassThroughEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerMismatchedEmbedder
from allennlp.nn.initializers import XavierNormalInitializer, OrthogonalInitializer, InitializerApplicator
from allennlp.training import Trainer, Checkpointer, GradientDescentTrainer
from allennlp.training.learning_rate_schedulers import SlantedTriangular
from allennlp.training.optimizers import HuggingfaceAdamWOptimizer
from models.coref_type_ner import CoreferenceResolver# ConllCorefReader
from torch.utils.data import DataLoader
from readers.enlarge_read import ConllCorefReader
#from dataset_readers.conll import ConllCorefReader
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dirpath = os.path.abspath(os.getcwd())
train_filepath = os.path.join(dirpath, "conllTypes/train")
valid_filepath = os.path.join(dirpath, "conllTypes/dev")
transformer_model = "SpanBERT/spanbert-base-cased"
max_length = 512
feature_size = 20
max_span_width = 30
transformer_dim = 768  # uniquely determined by transformer_model
span_embedding_dim = 3*transformer_dim + feature_size*2
span_pair_embedding_dim = 3*span_embedding_dim + feature_size*2
token_indexer = PretrainedTransformerMismatchedIndexer(model_name=transformer_model, max_length=max_length)
reader = ConllCorefReader(max_span_width, {'bert_tokens': token_indexer}, max_sentences=110)
train_dataset = reader.read(train_filepath)
validation_dataset = reader.read(valid_filepath)

vocab = Vocabulary()
vocab.add_token_to_namespace('@@UNKNOWN@@','ner_tags')
vocab = vocab.from_instances(train_dataset+validation_dataset)
vocab.save_to_files(os.path.join(dirpath, "vocab/ner_type"))
train_dataset.index_with(vocab)
validation_dataset.index_with(vocab)
for instance in train_dataset.instances:
    instance.index_fields(vocab)
for instance in validation_dataset.instances:
    instance.index_fields(vocab)
# train_dataset.instances[1].fields['pos_tags'].index(vocab)
ner_tag_count = vocab.get_vocab_size("ner_tags")
# Construct a dataloader directly for a dataset which contains allennlp
# Instances which have _already_ been indexed.
train_sampler = BucketBatchSampler(train_dataset, batch_size=1, sorting_keys=["text"], padding_noise=0.0)
train_loader = DataLoader(train_dataset, batch_size=1, batch_sampler=train_sampler, collate_fn=allennlp_collate)
dev_sampler = BucketBatchSampler(validation_dataset, batch_size=1, sorting_keys=["text"], padding_noise=0.0)
dev_loader = DataLoader(validation_dataset, batch_size=1, batch_sampler=dev_sampler, collate_fn=allennlp_collate)
embedding = PretrainedTransformerMismatchedEmbedder(transformer_model, max_length=max_length)
embedder = BasicTextFieldEmbedder(token_embedders={'bert_tokens': embedding})
encoder = PassThroughEncoder(input_dim=transformer_dim)
mention_feedforward = FeedForward(span_embedding_dim, 2, [1500, 1500], torch.nn.ReLU(), dropout=0.3)
antecedent_feedforward = FeedForward(span_pair_embedding_dim, 2, [1500, 1500], torch.nn.ReLU(), dropout=0.3)
normal_initial = XavierNormalInitializer()
orthogonal_initial = OrthogonalInitializer()
initial_para = [[".*_span_updating_gated_sum.*weight", normal_initial],
                [".*linear_layers.*weight", normal_initial],
                [".*scorer.*weight", normal_initial],
                ["_distance_embedding.weight", normal_initial],
                ["_span_width_embedding.weight", normal_initial],
                ["_context_layer._module.weight_ih.*", normal_initial],
                ["_context_layer._module.weight_hh.*", orthogonal_initial]
              ]
initializer = InitializerApplicator(regexes=initial_para)
corefer = CoreferenceResolver(vocab, text_field_embedder=embedder, context_layer=encoder,
                              mention_feedforward=mention_feedforward, antecedent_feedforward=antecedent_feedforward,
                              feature_size=feature_size, max_span_width=max_span_width, spans_per_word=0.4,
                              max_antecedents=100, span_types_size=39, ner_size=ner_tag_count, coarse_to_fine=True,
                              inference_order=2, lexical_dropout=0.5).to(device)
def build_trainer(
    model: Model,
    serialization_dir: str,
    train_loader: DataLoader,
    dev_loader: DataLoader
) -> Trainer:
    parameters = [
        (n, p)
        for n, p in model.named_parameters()
    ]

    grouppara = [
        [[".*transformer.*"], {"lr": 1e-5}]
    ]
    optimizer = HuggingfaceAdamWOptimizer(parameters, grouppara, lr=4e-4)
    learning_rate_scheduler = SlantedTriangular(optimizer, num_epochs=40, cut_frac=0.06)
    checkpoint = Checkpointer(serialization_dir=serialization_dir)
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        checkpointer=checkpoint,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        validation_metric="+coref_f1",
        patience=20,
        num_epochs=40,
        cuda_device=device,
        learning_rate_scheduler=learning_rate_scheduler,
        optimizer=optimizer
    )
    return trainer

serialization_dir = os.path.join(dirpath, "spanbert_ner_type_check")
#history_dir = os.path.join(dirpath, "models")
#with open(os.path.join(history_dir, 'conll_base.th'), 'rb') as f:
#     corefer.load_state_dict(torch.load(f))
trainer = build_trainer(
    corefer,
    serialization_dir,
    train_loader,
    dev_loader
)

print("Starting training")
trainer.train()
print("Finished training")
