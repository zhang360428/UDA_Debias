import json
import copy
import random
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel
import os
from matplotlib.ticker import MaxNLocator

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

model_map = {
    'loli:gpt-4o-2024-08-06': 0,
    'loli:deepseek-chat': 1,
    'loli:claude-3-5-sonnet-20241022': 2,
    'loli:glm-4-plus': 3,
    'loli:glm-4-air': 4,
    'loli:glm-4-flash': 5,
    'loli:doubao-1.5-pro-32k-250115': 6,
    'loli:qwen-max-2025-01-25': 7,
    'loli:gemini-2.0-flash': 8,
    'loli:deepseek-reasoner': 9
}
reverse_model_map = {v: k for k, v in model_map.items()}
result_map = {'A': 0, 'B': 1, 'C': 2}


class GroupedRandomSampler(Sampler):
    def __init__(self, data_source, group_key='judge_model', shuffle=True):
        self.data_source = data_source
        self.group_key = group_key
        self.shuffle = shuffle

        # 构建分组字典：group -> list of indices
        self.groups = defaultdict(list)
        for idx, item in enumerate(data_source):
            group = item[self.group_key]
            self.groups[group].append(idx)

        self.group_keys = list(self.groups.keys())

    def __iter__(self):
        indices = []
        for group in self.group_keys:
            group_indices = self.groups[group]
            if self.shuffle:
                group_indices = random.sample(group_indices, len(group_indices))
            indices.extend(group_indices)
        return iter(indices)

    def __len__(self):
        return len(self.data_source)

    def set_epoch(self, epoch):
        pass

class OptimizedModelAnswerDataset(Dataset):
    def __init__(self, data_dir):
        self.data = []
        self.models = []
        self.tokenizer = BertTokenizer.from_pretrained('./Models/bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('./Models/bert-base-uncased')
        self.load_data(data_dir)

    def process_text(self, text):
        return self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

    def compute_embedding(self, tokens):
        with torch.no_grad():
            embedding = self.bert_model(**tokens).last_hidden_state.mean(dim=1).squeeze()
        return embedding

    def load_data(self, data_dir):
        tokenizer_cache = {}
        embedding_cache = {}
        device = torch.device('cpu')
        self.bert_model = self.bert_model.to(device)

        file_list = sorted(os.listdir(data_dir))
        for filename in tqdm(file_list, desc="Loading data"):
            if filename.endswith(".json"):
                file_path = os.path.join(data_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    for item in json_data:
                        tokenizer_cache.clear()
                        embedding_cache.clear()
                        answers_dict = {ans['model']: ans['answer'] for ans in item['answers']}
                        for analysis in item['analysis']:
                            qid = item['qid']
                            model1 = analysis['model_1']
                            model2 = analysis['model_2']
                            result = analysis['result']
                            human_result = analysis['human_result']
                            judge_model = analysis['judge_model']

                            answer1 = answers_dict.get(model1, "")
                            answer2 = answers_dict.get(model2, "")
                            judge_answer = answers_dict.get(judge_model, "")

                            def process_text_with_cache(text):
                                if text not in tokenizer_cache:
                                    tokens = self.process_text(text)
                                    tokenizer_cache[text] = tokens
                                    tokens_device = {k: v.to(device) for k, v in tokens.items()}
                                    embedding = self.compute_embedding(tokens_device)
                                    embedding_cache[text] = embedding.cpu()
                                return tokenizer_cache[text], embedding_cache[text]

                            answer1_tokens, answer1_embedding = process_text_with_cache(answer1)
                            answer2_tokens, answer2_embedding = process_text_with_cache(answer2)
                            judge_tokens, judge_embedding = process_text_with_cache(judge_answer)

                            self.data.append({
                                'qid': qid,
                                'model1': model1,
                                'model2': model2,
                                'judge_model': judge_model,
                                'result': result,
                                'human_result': human_result,
                                'answer1_input_ids': answer1_tokens['input_ids'].squeeze(),
                                'answer1_attention_mask': answer1_tokens['attention_mask'].squeeze(),
                                'answer1_embedding': answer1_embedding,
                                'answer2_input_ids': answer2_tokens['input_ids'].squeeze(),
                                'answer2_attention_mask': answer2_tokens['attention_mask'].squeeze(),
                                'answer2_embedding': answer2_embedding,
                                'judge_answer_input_ids': judge_tokens['input_ids'].squeeze(),
                                'judge_answer_attention_mask': judge_tokens['attention_mask'].squeeze(),
                                'judge_answer_embedding': judge_embedding,
                            })

                            if model1 not in self.models:
                                self.models.append(model1)
                            if model2 not in self.models:
                                self.models.append(model2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'qid': item['qid'],
            'model1': model_map.get(item['model1']),
            'model2': model_map.get(item['model2']),
            'judge_model': model_map.get(item['judge_model']),
            'result': result_map.get(item['result']),
            'human_result': result_map.get(item['human_result']),
            'answer1_input_ids': item['answer1_input_ids'],
            'answer1_attention_mask': item['answer1_attention_mask'],
            'answer1_embedding': item['answer1_embedding'],
            'answer2_input_ids': item['answer2_input_ids'],
            'answer2_attention_mask': item['answer2_attention_mask'],
            'answer2_embedding': item['answer2_embedding'],
            'judge_answer_input_ids': item['judge_answer_input_ids'],
            'judge_answer_attention_mask': item['judge_answer_attention_mask'],
            'judge_answer_embedding': item['judge_answer_embedding'],
        }


class OptimizedEloSystem:
    def __init__(self, base_rating=1200.0):
        self.base_rating = base_rating
        self.ratings = defaultdict(lambda: base_rating)

    def update(self, model1, model2, result, K=32):
        r1 = self.ratings[model1]
        r2 = self.ratings[model2]

        e1 = 1 / (1 + torch.e ** ((r2 - r1) / 400))
        e2 = 1 - e1

        if result == "A":
            s1, s2 = 1, 0
        elif result == "B":
            s1, s2 = 0, 1
        elif result == "C":
            s1, s2 = 0.5, 0.5

        self.ratings[model1] = r1 + K * (s1 - e1)
        self.ratings[model2] = r2 + K * (s2 - e2)

    def get_ratings(self):
        return dict(self.ratings)


class OptimizedImprovedEloSystem(OptimizedEloSystem):
    def __init__(self, base_rating=1200.0):
        super().__init__(base_rating)
        self.ratings = defaultdict(
            lambda: torch.tensor([base_rating], dtype=torch.float32, requires_grad=True).to(torch.device('cpu')))

    def dynamic_update(self, model1, model2, result, answer1_embedding, answer2_embedding, judge_embedding=None,
                       adjusted_K=1, adjusted_s1=1, adjusted_s2=1, ifprint=False):
        device = torch.device('cpu')
        K = torch.tensor([32.0], dtype=torch.float32).to(device)
        similarity = torch.nn.functional.cosine_similarity(answer1_embedding.unsqueeze(0),
                                                           answer2_embedding.unsqueeze(0), dim=1)
        result_k = K * (1 - similarity) * adjusted_K

        r1 = self.ratings[model1]
        r2 = self.ratings[model2]

        if judge_embedding is not None:
            diff1 = F.cosine_similarity(answer1_embedding.unsqueeze(0),
                                        judge_embedding.unsqueeze(0), dim=1)
            diff2 = F.cosine_similarity(answer2_embedding.unsqueeze(0),
                                        judge_embedding.unsqueeze(0), dim=1)

            weights = torch.stack([diff1, diff2])
            weights = F.softmax(weights, dim=0)
            weight = weights[0]

            if result == "A":
                s1 = 1 + (1 - weight) * adjusted_s1
                s2 = weight * adjusted_s2
            elif result == "B":
                s1 = (1 - weight) * adjusted_s1
                s2 = 1 + weight * adjusted_s2
            elif result == "C":
                s1 = 0.5 + (1 - weight) * adjusted_s1
                s2 = 0.5 + weight * adjusted_s2

            eps = 1e-16
            total = s1 + s2 + eps
            s1 = s1 / total
            s2 = 1 - s1

            e1 = (1 / (1 + torch.exp((r2 - r1) / 400.0))).to(device)
            e2 = (1 - e1).to(device)

            if ifprint:
                print(f"model1: {model1}, ratings: {self.ratings[model1]}, "
                      f"r1: {r1}, e1: {e1}, s1: {s1}, result_k: {result_k}")
                print(f"device of model1: {model1}: {self.ratings[model1].device}")
                print(f"device of r1: {r1.device}")
                print(f"device of e1: {e1.device}")
                print(f"device of s1: {s1.device}")
                print(f"device of result_k: {result_k.device}")
            self.ratings[model1] = r1 + result_k * (s1 - e1)
            self.ratings[model2] = r2 + result_k * (s2 - e2)

    def get_ratings(self):
        return {k: v.detach() for k, v in self.ratings.items()}


class CoefficientLearner(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc2 = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 2048),
            nn.LayerNorm(2048),
            nn.LeakyReLU(0.1),
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 3)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.fc2(x)
        k_factor_coeff = F.sigmoid(x[:, 0]) * 30.0
        s1_coeff = F.sigmoid(x[:, 1]) * 30.0
        s2_coeff = F.sigmoid(x[:, 2]) * 30.0
        return k_factor_coeff, s1_coeff, s2_coeff


def train_coefficient_learner(model, train_dataset, val_dataset, epochs=80, batch_size=630):
    device = torch.device('cpu')
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=4e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    val_dataloader = DataLoader(val_dataset, batch_size=8100, shuffle=False)
    biggest_val_loss = float('inf')

    tmp_model = copy.deepcopy(model)

    train_losses = []
    val_losses = []

    def pearson_corr_loss(y_true, y_pred):
        y_true_mean = torch.mean(y_true)
        y_pred_mean = torch.mean(y_pred)
        numerator = torch.sum((y_true - y_true_mean) * (y_pred - y_pred_mean))
        denominator = (torch.sqrt(torch.sum((y_true - y_true_mean) ** 2)) *
                       torch.sqrt(torch.sum((y_pred - y_pred_mean) ** 2)) + 1e-8)
        corr = numerator / denominator
        return 1 - corr


    for epoch in range(epochs):
        g = torch.Generator()
        g.seed()

        train_sampler = GroupedRandomSampler(train_dataset, group_key='judge_model', shuffle=True)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=18900,
            sampler=train_sampler,
            generator=g
        )

        model.train()
        total_loss = 0.0
        improved_scores_batches = []
        traditional_scores_batches = []

        optimizer.zero_grad()
        for batch in train_dataloader:
            improved_elo = OptimizedImprovedEloSystem()
            traditional_elo = OptimizedEloSystem()
            answer1_embeddings = batch['answer1_embedding'].to(device)
            answer2_embeddings = batch['answer2_embedding'].to(device)
            judge_answer_embeddings = batch['judge_answer_embedding'].to(device)

            embeddings_diff = torch.abs(answer1_embeddings - answer2_embeddings)
            embeddings_dot = answer1_embeddings * answer2_embeddings
            judge_diff1 = torch.abs(answer1_embeddings - judge_answer_embeddings)
            judge_dot1 = answer1_embeddings * judge_answer_embeddings
            judge_diff2 = torch.abs(answer2_embeddings - judge_answer_embeddings)
            judge_dot2 = answer2_embeddings * judge_answer_embeddings
            similarity_answer = F.cosine_similarity(answer1_embeddings, answer2_embeddings).unsqueeze(1)
            similarity_judge1 = F.cosine_similarity(answer1_embeddings, judge_answer_embeddings).unsqueeze(1)
            similarity_judge2 = F.cosine_similarity(answer2_embeddings, judge_answer_embeddings).unsqueeze(1)

            answer1_probs = F.softmax(answer1_embeddings, dim=1)
            answer2_probs = F.softmax(answer2_embeddings, dim=1)
            judge_answer_probs = F.softmax(judge_answer_embeddings, dim=1)
            kl_answer1_answer2 = F.kl_div(torch.log(answer1_probs), answer2_probs, reduction='none').sum(dim=1,
                                                                                                         keepdim=True)
            kl_judge_answer1 = F.kl_div(torch.log(judge_answer_probs), answer1_probs, reduction='none').sum(dim=1,
                                                                                                            keepdim=True)
            kl_judge_answer2 = F.kl_div(torch.log(judge_answer_probs), answer2_probs, reduction='none').sum(dim=1,
                                                                                                            keepdim=True)

            answer1_norm = torch.norm(answer1_embeddings, dim=1, keepdim=True)
            answer2_norm = torch.norm(answer2_embeddings, dim=1, keepdim=True)
            judge_norm = torch.norm(judge_answer_embeddings, dim=1, keepdim=True)
            norm_diff = torch.abs(answer1_norm - answer2_norm)
            norm_diff1 = torch.abs(answer1_norm - judge_norm)
            norm_diff2 = torch.abs(answer2_norm - judge_norm)

            embedding_dot_norm = embeddings_dot / (answer1_norm * answer2_norm + 1e-8)
            embedding_dot_norm1 = judge_dot1 / (answer1_norm * judge_norm + 1e-8)
            embedding_dot_norm2 = judge_dot2 / (answer2_norm * judge_norm + 1e-8)

            inputs = torch.cat(
                [embeddings_diff, judge_diff1, judge_diff2, similarity_answer, similarity_judge1, similarity_judge2,
                 kl_answer1_answer2, kl_judge_answer1, kl_judge_answer2, norm_diff, norm_diff1, norm_diff2,
                 embedding_dot_norm,
                 embedding_dot_norm1, embedding_dot_norm2], dim=1)

            k_coeffs, s1_coeffs, s2_coeffs = model(inputs)

            batch_size = len(batch['model1'])

            for m in model_map.keys():
                improved_elo.ratings[m] = torch.tensor([1200.0], dtype=torch.float32, requires_grad=True, device=device)
                traditional_elo.ratings[m] = 1200.0  # m是模型名称

            for i in range(batch_size):
                model1 = reverse_model_map[batch['model1'][i].item()]
                model2 = reverse_model_map[batch['model2'][i].item()]
                result = list(result_map.keys())[batch['result'][i].item()]

                improved_elo.dynamic_update(
                    model1, model2, result,
                    answer1_embeddings[i], answer2_embeddings[i],
                    judge_answer_embeddings[i],
                    k_coeffs[i], s1_coeffs[i], s2_coeffs[i],
                    ifprint=False
                )
                traditional_elo.update(model1, model2, result)

            improved_scores = []
            traditional_scores = []

            for m in model_map.keys():
                improved_scores.append(improved_elo.ratings[m])
                traditional_scores.append(torch.tensor([traditional_elo.ratings[m]], device=device))

            improved_scores = torch.cat(improved_scores)
            traditional_scores = torch.cat(traditional_scores)
            improved_scores_batches.append(improved_scores)
            traditional_scores_batches.append(traditional_scores)

        all_ratings = torch.stack(improved_scores_batches)
        traditional_ratings = torch.stack(traditional_scores_batches)
        rating_mean = torch.mean(all_ratings, dim=0, keepdim=True).squeeze(0)
        global_mean = torch.mean(traditional_ratings, dim=0, keepdim=True).squeeze(0)

        alpha = 0.1
        beta = 1000.0
        sigma = 0.2
        consistency_loss_value = alpha * F.mse_loss(improved_scores_batches[0], global_mean) + beta * pearson_corr_loss(improved_scores_batches[0], global_mean)
        consistency_loss_value += F.mse_loss(rating_mean, global_mean) * sigma

        for i in range(1, len(improved_scores_batches)):
            xa = alpha * F.mse_loss(improved_scores_batches[i], global_mean)
            xb = beta * pearson_corr_loss(improved_scores_batches[i], global_mean)
            consistency_loss_value += xa + xb


        consistency_loss_value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

        train_loss = consistency_loss_value.item()
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            improved_scores_batches = []
            traditional_scores_batches = []

            for batch in val_dataloader:
                improved_elo = OptimizedImprovedEloSystem()
                traditional_elo = OptimizedEloSystem()
                answer1_embeddings = batch['answer1_embedding'].to(device)
                answer2_embeddings = batch['answer2_embedding'].to(device)
                judge_answer_embeddings = batch['judge_answer_embedding'].to(device)

                embeddings_diff = torch.abs(answer1_embeddings - answer2_embeddings)
                embeddings_dot = answer1_embeddings * answer2_embeddings
                judge_diff1 = torch.abs(answer1_embeddings - judge_answer_embeddings)
                judge_dot1 = answer1_embeddings * judge_answer_embeddings
                judge_diff2 = torch.abs(answer2_embeddings - judge_answer_embeddings)
                judge_dot2 = answer2_embeddings * judge_answer_embeddings
                similarity_answer = F.cosine_similarity(answer1_embeddings, answer2_embeddings).unsqueeze(1)
                similarity_judge1 = F.cosine_similarity(answer1_embeddings, judge_answer_embeddings).unsqueeze(1)
                similarity_judge2 = F.cosine_similarity(answer2_embeddings, judge_answer_embeddings).unsqueeze(1)

                answer1_probs = F.softmax(answer1_embeddings, dim=1)
                answer2_probs = F.softmax(answer2_embeddings, dim=1)
                judge_answer_probs = F.softmax(judge_answer_embeddings, dim=1)
                kl_answer1_answer2 = F.kl_div(torch.log(answer1_probs), answer2_probs, reduction='none').sum(dim=1,
                                                                                                             keepdim=True)
                kl_judge_answer1 = F.kl_div(torch.log(judge_answer_probs), answer1_probs, reduction='none').sum(dim=1,
                                                                                                                keepdim=True)
                kl_judge_answer2 = F.kl_div(torch.log(judge_answer_probs), answer2_probs, reduction='none').sum(dim=1,
                                                                                                                keepdim=True)

                answer1_norm = torch.norm(answer1_embeddings, dim=1, keepdim=True)
                answer2_norm = torch.norm(answer2_embeddings, dim=1, keepdim=True)
                judge_norm = torch.norm(judge_answer_embeddings, dim=1, keepdim=True)
                norm_diff = torch.abs(answer1_norm - answer2_norm)
                norm_diff1 = torch.abs(answer1_norm - judge_norm)
                norm_diff2 = torch.abs(answer2_norm - judge_norm)
                embedding_dot_norm = embeddings_dot / (answer1_norm * answer2_norm + 1e-8)
                embedding_dot_norm1 = judge_dot1 / (answer1_norm * judge_norm + 1e-8)
                embedding_dot_norm2 = judge_dot2 / (answer2_norm * judge_norm + 1e-8)

                inputs = torch.cat(
                    [embeddings_diff, judge_diff1, judge_diff2, similarity_answer, similarity_judge1, similarity_judge2,
                     kl_answer1_answer2, kl_judge_answer1, kl_judge_answer2, norm_diff, norm_diff1, norm_diff2,
                     embedding_dot_norm, embedding_dot_norm1, embedding_dot_norm2], dim=1)

                k_coeffs, s1_coeffs, s2_coeffs = model(inputs)

                batch_size = len(batch['model1'])
                for m in model_map.keys():
                    improved_elo.ratings[m] = torch.tensor([1200.0], dtype=torch.float32, requires_grad=False,
                                                           device=device)
                    traditional_elo.ratings[m] = 1200.0
                for i in range(batch_size):
                    model1 = reverse_model_map[batch['model1'][i].item()]
                    model2 = reverse_model_map[batch['model2'][i].item()]
                    result = list(result_map.keys())[batch['result'][i].item()]

                    improved_elo.dynamic_update(
                        model1, model2, result,
                        answer1_embeddings[i], answer2_embeddings[i],
                        judge_answer_embeddings[i],
                        k_coeffs[i], s1_coeffs[i], s2_coeffs[i],
                        ifprint=False
                    )
                    traditional_elo.update(model1, model2, result)

                improved_scores = []
                traditional_scores = []
                for m in model_map.keys():
                    improved_scores.append(improved_elo.ratings[m])
                    traditional_scores.append(torch.tensor([traditional_elo.ratings[m]], device=device))

                improved_scores = torch.cat(improved_scores)
                traditional_scores = torch.cat(traditional_scores)
                improved_scores_batches.append(improved_scores)
                traditional_scores_batches.append(traditional_scores)

            all_ratings = torch.stack(improved_scores_batches)
            traditional_ratings = torch.stack(traditional_scores_batches)
            rating_mean = torch.mean(all_ratings, dim=0, keepdim=True).squeeze(0)
            global_mean = torch.mean(traditional_ratings, dim=0, keepdim=True).squeeze(0)

            consistency_loss_value = alpha * F.mse_loss(improved_scores_batches[0], global_mean) + beta * pearson_corr_loss(improved_scores_batches[0], global_mean)
            consistency_loss_value += F.mse_loss(rating_mean, global_mean) * 0.2

            for i in range(1, len(improved_scores_batches)):
                xa = alpha * F.mse_loss(improved_scores_batches[i], global_mean)
                xb = beta * pearson_corr_loss(improved_scores_batches[i], global_mean)
                consistency_loss_value += xa + xb


            val_loss = consistency_loss_value.item()
            val_losses.append(val_loss)

        avg_val_loss = val_loss
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < biggest_val_loss:
            biggest_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model_arenaxxxxx.pth')
            print(f"Saving model with validation loss: {biggest_val_loss:.4f}")
            tmp_model = copy.deepcopy(model)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs_range = range(1, epochs + 1)
    ax.plot(epochs_range, train_losses, color='#1f77b4', linewidth=2, label='Training Loss', marker='o', markersize=4)
    ax.plot(epochs_range, val_losses, color='#ff7f0e', linewidth=2, label='Validation Loss', marker='s', markersize=4)

    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax.set_title('Training and Validation Loss Curves', fontsize=16, fontweight='bold', pad=20)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle='--', alpha=0.7)

    ax.legend(frameon=True, fancybox=True, shadow=True, loc='upper right', fontsize=12)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig('xxxxx.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogy(epochs_range, train_losses, color='#1f77b4', linewidth=2, label='Training Loss', marker='o',
                markersize=4)
    ax.semilogy(epochs_range, val_losses, color='#ff7f0e', linewidth=2, label='Validation Loss', marker='s',
                markersize=4)

    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss (log scale)', fontsize=14, fontweight='bold')
    ax.set_title('Training and Validation Loss Curves (Log Scale)', fontsize=16, fontweight='bold', pad=20)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(frameon=True, fancybox=True, shadow=True, loc='upper right', fontsize=12)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig('xxxxx.png', dpi=300, bbox_inches='tight')
    plt.close()

    return tmp_model


def calculate_human_elo(dataset, models):
    elo = OptimizedEloSystem()
    for item in dataset.data:
        model1_idx = item['model1']
        model2_idx = item['model2']
        human_result_idx = item['human_result']
        elo.update(model1_idx, model2_idx, human_result_idx)

    human_elo = {model: elo.ratings.get(model, 1200) for model in models}
    return human_elo


def calculate_model_elo(dataset, models):
    elo = OptimizedEloSystem()
    for item in dataset.data:
        model1_idx = item['model1']
        model2_idx = item['model2']
        result_idx = item['result']
        elo.update(model1_idx, model2_idx, result_idx)

    model_elo = {model: elo.ratings.get(model, 1200) for model in models}
    return model_elo

def calculate_improved_elo(dataset, models, learner_model):
    improved_elo = OptimizedImprovedEloSystem()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device('cpu')
    learner_model = learner_model.to(device)

    with tqdm(dataloader, unit="batch") as loop:
        for batch in loop:
            item = batch
            model1_idx = item['model1'].item()
            model2_idx = item['model2'].item()
            result_idx = item['result'].item()
            model1 = reverse_model_map.get(model1_idx, "")
            model2 = reverse_model_map.get(model2_idx, "")
            result = list(result_map.keys())[list(result_map.values()).index(result_idx)]

            answer1_embeddings = batch['answer1_embedding'].to(device)
            answer2_embeddings = batch['answer2_embedding'].to(device)
            judge_answer_embeddings = batch['judge_answer_embedding'].to(device)

            embeddings_diff = torch.abs(answer1_embeddings - answer2_embeddings)
            embeddings_dot = answer1_embeddings * answer2_embeddings
            judge_diff1 = torch.abs(answer1_embeddings - judge_answer_embeddings)
            judge_dot1 = answer1_embeddings * judge_answer_embeddings
            judge_diff2 = torch.abs(answer2_embeddings - judge_answer_embeddings)
            judge_dot2 = answer2_embeddings * judge_answer_embeddings
            similarity_answer = F.cosine_similarity(answer1_embeddings, answer2_embeddings).unsqueeze(1)
            similarity_judge1 = F.cosine_similarity(answer1_embeddings, judge_answer_embeddings).unsqueeze(1)
            similarity_judge2 = F.cosine_similarity(answer2_embeddings, judge_answer_embeddings).unsqueeze(1)

            answer1_probs = F.softmax(answer1_embeddings, dim=1)
            answer2_probs = F.softmax(answer2_embeddings, dim=1)
            judge_answer_probs = F.softmax(judge_answer_embeddings, dim=1)
            kl_answer1_answer2 = F.kl_div(torch.log(answer1_probs), answer2_probs, reduction='none').sum(dim=1,
                                                                                                         keepdim=True)
            kl_judge_answer1 = F.kl_div(torch.log(judge_answer_probs), answer1_probs, reduction='none').sum(dim=1,
                                                                                                            keepdim=True)
            kl_judge_answer2 = F.kl_div(torch.log(judge_answer_probs), answer2_probs, reduction='none').sum(dim=1,
                                                                                                            keepdim=True)

            answer1_norm = torch.norm(answer1_embeddings, dim=1, keepdim=True)
            answer2_norm = torch.norm(answer2_embeddings, dim=1, keepdim=True)
            judge_norm = torch.norm(judge_answer_embeddings, dim=1, keepdim=True)
            norm_diff = torch.abs(answer1_norm - answer2_norm)
            norm_diff1 = torch.abs(answer1_norm - judge_norm)
            norm_diff2 = torch.abs(answer2_norm - judge_norm)
            embedding_dot_norm = embeddings_dot / (answer1_norm * answer2_norm + 1e-8)
            embedding_dot_norm1 = judge_dot1 / (answer1_norm * judge_norm + 1e-8)
            embedding_dot_norm2 = judge_dot2 / (answer2_norm * judge_norm + 1e-8)

            inputs = torch.cat(
                [embeddings_diff, judge_diff1, judge_diff2, similarity_answer, similarity_judge1, similarity_judge2,
                 kl_answer1_answer2, kl_judge_answer1, kl_judge_answer2, norm_diff, norm_diff1, norm_diff2,
                 embedding_dot_norm,
                 embedding_dot_norm1, embedding_dot_norm2], dim=1)

            with torch.no_grad():
                k_coeffs, s1_coeffs, s2_coeffs = learner_model(inputs)

            improved_elo.dynamic_update(
                model1, model2, result,
                answer1_embeddings.squeeze(),
                answer2_embeddings.squeeze(),
                judge_answer_embeddings.squeeze(),
                adjusted_K=k_coeffs,
                adjusted_s1=s1_coeffs,
                adjusted_s2=s2_coeffs,
                ifprint=False
            )

    improved_elo_scores = {model: improved_elo.ratings.get(model, 1200) for model in models}
    print("improved_elo_scores:", improved_elo_scores)
    return improved_elo_scores


def calculate_human_elo_grouped(dataset, models):
    elo_grouped = defaultdict(lambda: OptimizedEloSystem())
    judge_models = list(set(item['judge_model'] for item in dataset.data))

    for item in dataset.data:
        judge_model = item['judge_model']
        model1_idx = item['model1']
        model2_idx = item['model2']
        human_result_idx = item['human_result']
        elo_grouped[judge_model].update(model1_idx, model2_idx, human_result_idx)

    human_elo_grouped = {}
    for judge_model in judge_models:
        human_elo = {model: elo_grouped[judge_model].ratings.get(model, 1200) for model in models}
        human_elo_grouped[judge_model] = human_elo
    return human_elo_grouped


def calculate_model_elo_grouped(dataset, models):
    elo_grouped = defaultdict(lambda: OptimizedEloSystem())
    judge_models = list(set(item['judge_model'] for item in dataset.data))

    for item in dataset.data:
        judge_model = item['judge_model']
        model1_idx = item['model1']
        model2_idx = item['model2']
        result_idx = item['result']
        elo_grouped[judge_model].update(model1_idx, model2_idx, result_idx)

    model_elo_grouped = {}
    for judge_model in judge_models:
        model_elo = {model: elo_grouped[judge_model].ratings.get(model, 1200) for model in models}
        model_elo_grouped[judge_model] = model_elo
    return model_elo_grouped


def calculate_improved_elo_grouped(dataset, models, learner_model):
    elo_grouped = defaultdict(lambda: OptimizedImprovedEloSystem())
    judge_models = list(set(item['judge_model'] for item in dataset.data))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device('cpu')
    learner_model = learner_model.to(device)

    for batch in dataloader:
        item = batch
        judge_model = item['judge_model'].item()
        model1_idx = item['model1'].item()
        model2_idx = item['model2'].item()
        result_idx = item['result'].item()
        model1 = reverse_model_map.get(model1_idx, "")
        model2 = reverse_model_map.get(model2_idx, "")
        result = list(result_map.keys())[list(result_map.values()).index(result_idx)]

        answer1_embedding = item['answer1_embedding'].to(device)
        answer2_embedding = item['answer2_embedding'].to(device)
        judge_answer_embedding = item['judge_answer_embedding'].to(device)

        embeddings_diff = torch.abs(answer1_embedding - answer2_embedding)
        embeddings_dot = answer1_embedding * answer2_embedding
        judge_diff1 = torch.abs(answer1_embedding - judge_answer_embedding)
        judge_dot1 = answer1_embedding * judge_answer_embedding
        judge_diff2 = torch.abs(answer2_embedding - judge_answer_embedding)
        judge_dot2 = answer2_embedding * judge_answer_embedding
        inputs = torch.cat([embeddings_diff, embeddings_dot, judge_diff1, judge_diff2, judge_dot1, judge_dot2], dim=1)

        with torch.no_grad():
            k_coeffs, s1_coeffs, s2_coeffs = learner_model(inputs)

        adjusted_k = k_coeffs.item()
        adjusted_s1 = s1_coeffs.item()
        adjusted_s2 = s2_coeffs.item()

        elo_grouped[judge_model].dynamic_update(
            model1, model2, result,
            answer1_embedding.squeeze(),
            answer2_embedding.squeeze(),
            judge_answer_embedding.squeeze(),
            adjusted_K=adjusted_k,
            adjusted_s1=adjusted_s1,
            adjusted_s2=adjusted_s2,
            ifprint=False
        )

    improved_elo_grouped = {}
    for judge_model in judge_models:
        improved_elo = {}
        judge_model_idx = model_map.get(judge_model, -1)
        for model in models:
            rating = elo_grouped[judge_model_idx].ratings[model].item() if isinstance(
                elo_grouped[judge_model_idx].ratings[model], torch.Tensor) else \
                elo_grouped[judge_model_idx].ratings[model]
            improved_elo[model] = rating
        improved_elo_grouped[judge_model] = improved_elo
    return improved_elo_grouped


def calculate_all_elo_grouped(dataset, models, learner_model):
    human_elo_grouped = defaultdict(lambda: OptimizedEloSystem())
    model_elo_grouped = defaultdict(lambda: OptimizedEloSystem())
    improved_elo_grouped = defaultdict(lambda: OptimizedImprovedEloSystem())
    judge_models = list(set(item['judge_model'] for item in dataset.data))
    judge_models.sort(key=lambda x: model_map.get(x, -1))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device('cpu')
    learner_model = learner_model.to(device)
    outputdata = []
    with open('./output/some_result.json', 'w') as f:
        for batch in dataloader:
            item = batch
            judge_model = item['judge_model'].item()
            model1_idx = item['model1'].item()
            model2_idx = item['model2'].item()
            result_idx = item['result'].item()
            human_result_idx = item['human_result'].item()
            model1 = reverse_model_map.get(model1_idx, "")
            model2 = reverse_model_map.get(model2_idx, "")
            result = list(result_map.keys())[list(result_map.values()).index(result_idx)]
            human_result = list(result_map.keys())[list(result_map.values()).index(human_result_idx)]

            answer1_embeddings = batch['answer1_embedding'].to(device)
            answer2_embeddings = batch['answer2_embedding'].to(device)
            judge_answer_embeddings = batch['judge_answer_embedding'].to(device)

            embeddings_diff = torch.abs(answer1_embeddings - answer2_embeddings)
            embeddings_dot = answer1_embeddings * answer2_embeddings
            judge_diff1 = torch.abs(answer1_embeddings - judge_answer_embeddings)
            judge_dot1 = answer1_embeddings * judge_answer_embeddings
            judge_diff2 = torch.abs(answer2_embeddings - judge_answer_embeddings)
            judge_dot2 = answer2_embeddings * judge_answer_embeddings
            similarity_answer = F.cosine_similarity(answer1_embeddings, answer2_embeddings).unsqueeze(1)
            similarity_judge1 = F.cosine_similarity(answer1_embeddings, judge_answer_embeddings).unsqueeze(1)
            similarity_judge2 = F.cosine_similarity(answer2_embeddings, judge_answer_embeddings).unsqueeze(1)

            answer1_probs = F.softmax(answer1_embeddings, dim=1)
            answer2_probs = F.softmax(answer2_embeddings, dim=1)
            judge_answer_probs = F.softmax(judge_answer_embeddings, dim=1)
            kl_answer1_answer2 = F.kl_div(torch.log(answer1_probs), answer2_probs, reduction='none').sum(dim=1,
                                                                                                         keepdim=True)
            kl_judge_answer1 = F.kl_div(torch.log(judge_answer_probs), answer1_probs, reduction='none').sum(dim=1,
                                                                                                            keepdim=True)
            kl_judge_answer2 = F.kl_div(torch.log(judge_answer_probs), answer2_probs, reduction='none').sum(dim=1,
                                                                                                            keepdim=True)

            answer1_norm = torch.norm(answer1_embeddings, dim=1, keepdim=True)
            answer2_norm = torch.norm(answer2_embeddings, dim=1, keepdim=True)
            judge_norm = torch.norm(judge_answer_embeddings, dim=1, keepdim=True)
            norm_diff = torch.abs(answer1_norm - answer2_norm)
            norm_diff1 = torch.abs(answer1_norm - judge_norm)
            norm_diff2 = torch.abs(answer2_norm - judge_norm)
            embedding_dot_norm = embeddings_dot / (answer1_norm * answer2_norm + 1e-8)
            embedding_dot_norm1 = judge_dot1 / (answer1_norm * judge_norm + 1e-8)
            embedding_dot_norm2 = judge_dot2 / (answer2_norm * judge_norm + 1e-8)

            inputs = torch.cat(
                [embeddings_diff, judge_diff1, judge_diff2, similarity_answer, similarity_judge1, similarity_judge2,
                 kl_answer1_answer2, kl_judge_answer1, kl_judge_answer2, norm_diff, norm_diff1, norm_diff2,
                 embedding_dot_norm,
                 embedding_dot_norm1, embedding_dot_norm2], dim=1)

            with torch.no_grad():
                k_coeffs, s1_coeffs, s2_coeffs = learner_model(inputs)

            adjusted_k = k_coeffs.item()
            adjusted_s1 = s1_coeffs.item()
            adjusted_s2 = s2_coeffs.item()

            human_elo_grouped[judge_model].update(
                model1, model2, human_result
            )
            model_elo_grouped[judge_model].update(
                model1, model2, result
            )
            improved_elo_grouped[judge_model].dynamic_update(
                model1, model2, result,
                answer1_embeddings.squeeze(),
                answer2_embeddings.squeeze(),
                judge_answer_embeddings.squeeze(),
                adjusted_K=adjusted_k,
                adjusted_s1=adjusted_s1,
                adjusted_s2=adjusted_s2,
                ifprint=False
            )
            outputdata.append({
                'qid': item['qid'],
                'model1': model1,
                'model2': model2,
                'judge_model': reverse_model_map.get(judge_model, ""),
                'result': result,
                'human_result': human_result,
                'human_elo_model1': human_elo_grouped[judge_model].ratings.get(model1, -1200),
                'human_elo_model2': human_elo_grouped[judge_model].ratings.get(model2, -1200),
                'model_elo_model1': model_elo_grouped[judge_model].ratings.get(model1, -1200),
                'model_elo_model2': model_elo_grouped[judge_model].ratings.get(model2, -1200),
            })

        json.dump(outputdata, f, ensure_ascii=False, indent=4)

    human_elo_grouped_dict = {}
    model_elo_grouped_dict = {}
    improved_elo_grouped_dict = {}
    for judge_model in judge_models:
        human_elo = {}
        model_elo = {}
        improved_elo = {}

        judge_model_idx = model_map.get(judge_model, -1)
        for model in models:
            rating = improved_elo_grouped[judge_model_idx].ratings[model].item() if isinstance(
                improved_elo_grouped[judge_model_idx].ratings[model], torch.Tensor) else \
                improved_elo_grouped[judge_model_idx].ratings[model]
            improved_elo[model] = rating
            human_elo[model] = human_elo_grouped[judge_model_idx].ratings.get(model, 1200)
            model_elo[model] = model_elo_grouped[judge_model_idx].ratings.get(model, 1200)
        human_elo_grouped_dict[judge_model] = human_elo
        model_elo_grouped_dict[judge_model] = model_elo
        improved_elo_grouped_dict[judge_model] = improved_elo
    return human_elo_grouped_dict, model_elo_grouped_dict, improved_elo_grouped_dict


def plot_results(human_elo, model_elo, improved_elo, models):
    plt.figure(figsize=(12, 6))
    human_values = [human_elo[model] for model in models]
    plt.bar(models, human_values)
    plt.xticks(rotation=45, ha='right')
    plt.title('Human Assessment ELO Scores')
    plt.ylabel('ELO Score')
    plt.tight_layout()
    plt.savefig('xxxxx.png')

    plt.figure(figsize=(12, 6))
    model_values = [model_elo[model] for model in models]
    plt.bar(models, model_values)
    plt.xticks(rotation=45, ha='right')
    plt.title('Model Assessment ELO Scores')
    plt.ylabel('ELO Score')
    plt.tight_layout()
    plt.savefig('xxxxx.png')

    # 条形图：纠正后的模型评估的ELO分数
    plt.figure(figsize=(12, 6))
    improved_values = [improved_elo[model] for model in models]
    plt.bar(models, torch.tensor(improved_values).cpu().numpy())
    plt.xticks(rotation=45, ha='right')
    plt.title('Improved Model Assessment ELO Scores')
    plt.ylabel('ELO Score')
    plt.tight_layout()
    plt.savefig('xxxxx.png')


def plot_grouped_results(human_elo_grouped, model_elo_grouped, improved_elo_grouped, models):
    judge_models = list(human_elo_grouped.keys())
    judge_models.sort(key=lambda kx: model_map.get(kx, -1))
    num_judge_models = len(judge_models)

    human_elo_matrix = np.zeros((num_judge_models, len(models)))
    model_elo_matrix = np.zeros((num_judge_models, len(models)))
    improved_elo_matrix = np.zeros((num_judge_models, len(models)))

    for i, judge_model in enumerate(judge_models):
        human_elo_matrix[i, :] = [human_elo_grouped[judge_model][model] for model in models]
        model_elo_matrix[i, :] = [model_elo_grouped[judge_model][model] for model in models]
        improved_elo_matrix[i, :] = [improved_elo_grouped[judge_model][model] for model in models]

    plt.figure(figsize=(10, 8))
    sns.heatmap(model_elo_matrix, xticklabels=models, yticklabels=judge_models, cmap='coolwarm', annot=False)
    plt.title('Original Model Assessment ELO Scores by Judge Model')
    plt.xlabel('Models')
    plt.ylabel('Judge Models')
    plt.tight_layout()
    plt.savefig('xxxxx.png')

    # 热力图：改进后的模型评估的ELO分数
    plt.figure(figsize=(10, 8))
    sns.heatmap(improved_elo_matrix, xticklabels=models, yticklabels=judge_models, cmap='coolwarm', annot=False)
    plt.title('Improved Model Assessment ELO Scores by Judge Model')
    plt.xlabel('Models')
    plt.ylabel('Judge Models')
    plt.tight_layout()
    plt.savefig('xxxxx.png')

    judge_model_names = judge_models

    original_correlation_pearson = []
    improved_correlation_pearson = []
    original_correlation_spearman = []
    improved_correlation_spearman = []

    human_groups = []
    original_groups = []
    improved_groups = []
    judge_groups = []

    for judge_model in judge_models:
        human_elo = human_elo_grouped[judge_model]
        model_elo = model_elo_grouped[judge_model]
        improved_elo = improved_elo_grouped[judge_model]

        human_values = []
        original_values = []
        improved_values = []
        for model in models:
            human_values.append(human_elo.get(model))
            original_values.append(model_elo.get(model))
            improved_values.append(improved_elo.get(model))

        print("human_values:", human_values)
        print("original_values:", original_values)
        print("improved_values:", improved_values)
        human_groups.append(human_values)
        original_groups.append(original_values)
        improved_groups.append(improved_values)
        judge_groups.append(judge_model)

        def compute_rank_with_ties(values):
            arr = np.array(values)
            sorted_indices = np.argsort(arr)
            sorted_values = arr[sorted_indices]

            unique_values, inverse_indices, counts = np.unique(
                sorted_values, return_inverse=True, return_counts=True
            )

            ranks = np.arange(1, len(arr) + 1)

            for i in range(len(unique_values)):
                if counts[i] > 1:
                    mask = (inverse_indices == i)
                    ranks[mask] = ranks[mask].mean()

            restored_ranks = np.empty_like(ranks)
            restored_ranks[sorted_indices] = ranks
            return restored_ranks

        def spearman_corr(tx, y):
            rank_x = compute_rank_with_ties(tx)
            rank_y = compute_rank_with_ties(y)
            return np.corrcoef(rank_x, rank_y)[0, 1]

        def person_corr(tx, y):
            x_mean = np.mean(tx)
            y_mean = np.mean(y)
            numerator = np.sum((tx - x_mean) * (y - y_mean))
            denominator = np.sqrt(np.sum((tx - x_mean) ** 2) * np.sum((y - y_mean) ** 2) + 1e-8)
            return numerator / denominator if denominator != 0 else 0

        corr_original_spearman = spearman_corr(human_values, original_values)
        corr_improved_spearman = spearman_corr(human_values, improved_values)

        corr_original_pearson = person_corr(np.array(human_values), np.array(original_values))
        corr_improved_pearson = person_corr(np.array(human_values), np.array(improved_values))
        print(f"corr_original_pearson is : {corr_original_pearson}")
        print(f"corr_improved_pearson is : {corr_improved_pearson}")
        original_correlation_pearson.append(corr_original_pearson)
        improved_correlation_pearson.append(corr_improved_pearson)
        original_correlation_spearman.append(corr_original_spearman)
        improved_correlation_spearman.append(corr_improved_spearman)

    print("human_groups:\n", human_groups)
    print("original_groups:\n", original_groups)
    print("improved_groups:\n", improved_groups)
    print("judge_groups:\n", judge_groups)

    plt.figure(figsize=(12, 6))
    x = np.arange(len(judge_model_names))
    plt.plot(x, original_correlation_pearson, marker='o', label='Original Model')
    plt.plot(x, improved_correlation_pearson, marker='o', label='Improved Model')

    plt.title('Correlation Coefficients with Human Assessment by Judge Model')
    plt.xlabel('Judge Models')
    plt.ylabel('Pearson Correlation Coefficient')
    plt.xticks(x, judge_model_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('xxxxx.png')

    # 相关性系数折线图
    plt.figure(figsize=(12, 6))
    x = np.arange(len(judge_model_names))
    plt.plot(x, original_correlation_spearman, marker='o', label='Original Model')
    plt.plot(x, improved_correlation_spearman, marker='o', label='Improved Model')

    plt.title('Correlation Coefficients with Human Assessment by Judge Model')
    plt.xlabel('Judge Models')
    plt.ylabel('Spearman Correlation Coefficient')
    plt.xticks(x, judge_model_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('xxxxx.png')


# 主函数
def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    os.environ["PYTHONHASHSEED"] = "42"
    data_dir = r"xxxxx"

    dataset = OptimizedModelAnswerDataset(data_dir)
    models = dataset.models

    models.sort(key=lambda x: model_map.get(x, -1))

    all_data = defaultdict(list)
    for item in dataset.data:
        key = (item['judge_model'] + item['model1'] + item['model2'])
        all_data[key].append(item)
    train_data = []
    test_data = []
    final_data = []
    for key, items in all_data.items():
        split_index = int(0.3 * len(items))
        final_data.extend(items)
        test_data.extend(items[:split_index])
        train_data.extend(items[split_index:])

    train_groups = defaultdict(list)
    for item in train_data:
        train_groups[item['judge_model'] + item['qid']].append(item)

    train_data = []
    for key, items in train_groups.items():
        train_data.extend(items)

    test_groups = defaultdict(list)
    for item in test_data:
        test_groups[item['judge_model'] + item['qid']].append(item)

    test_data = []
    for key, items in test_groups.items():
        test_data.extend(items)

    final_groups = defaultdict(list)
    for item in final_data:
        final_groups[item['judge_model'] + item['qid']].append(item)
    final_data = []
    for key, items in final_groups.items():
        final_data.extend(items)

    train_dataset = copy.deepcopy(dataset)
    train_dataset.data = train_data
    print(f"train_dataset.data length: {len(train_dataset.data)}")

    test_dataset = copy.deepcopy(dataset)
    test_dataset.data = test_data
    original_dataset = copy.deepcopy(dataset)
    original_dataset.data = test_data

    input_dim = 768 * 6 + 9
    learner_model = CoefficientLearner(input_dim)
    learner_model = train_coefficient_learner(learner_model, train_dataset, test_dataset)

    torch.save(learner_model.state_dict(), 'model_arenaxxxxx.pth')

    human_elo = calculate_human_elo(original_dataset, models)
    model_elo = calculate_model_elo(original_dataset, models)
    improved_elo = calculate_improved_elo(original_dataset, models, learner_model)

    plot_results(human_elo, model_elo, improved_elo, models)

    human_elo_grouped, model_elo_grouped, improved_elo_grouped = calculate_all_elo_grouped(original_dataset, models,
                                                                                           learner_model)

    plot_grouped_results(human_elo_grouped, model_elo_grouped, improved_elo_grouped, models)


if __name__ == "__main__":
    main()