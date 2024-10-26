import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig

class TextCNN(nn.Module):
    def __init__(self, kernel_initializer=None):
        super(TextCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=4, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.dropout = nn.Dropout(0.2)
        
        # Optional: Apply kernel_initializer if provided
        if kernel_initializer:
            self.apply(kernel_initializer)

    def forward(self, x):
        # Assuming x is of shape [batch_size, maxlen, 1]
        x = x.permute(0, 2, 1)  # Permute to shape [batch_size, 1, maxlen]
        
        cnn1 = self.conv1(x)
        cnn1 = F.relu(cnn1)
        cnn1 = F.adaptive_avg_pool1d(cnn1, 1).squeeze(-1)

        cnn2 = self.conv2(x)
        cnn2 = F.relu(cnn2)
        cnn2 = F.adaptive_avg_pool1d(cnn2, 1).squeeze(-1)

        cnn3 = self.conv3(x)
        cnn3 = F.relu(cnn3)
        cnn3 = F.adaptive_avg_pool1d(cnn3, 1).squeeze(-1)

        # Concatenate along the feature dimension
        output = torch.cat((cnn1, cnn2, cnn3), dim=1)
        output = self.dropout(output)

        return output
    
    class BertTextClassifier(nn.Module):
        def __init__(self, config_path, checkpoint_path, class_nums):
            super(BertTextClassifier, self).__init__()
            config = BertConfig.from_json_file(config_path)
            self.bert = BertModel.from_pretrained(checkpoint_path, config=config)
            self.textcnn = TextCNN(input_dim=config.hidden_size, output_dim=256)  # Example output dim
            self.fc = nn.Linear(config.hidden_size + 256, 512)
            self.classifier = nn.Linear(512, class_nums)
            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(dim=1)

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_features = outputs.last_hidden_state[:, 0]  # CLS token
            all_token_embedding = outputs.last_hidden_state[:, 1:-1]  # Excluding [CLS] and [SEP]

            cnn_features = self.textcnn(all_token_embedding)
            concat_features = torch.cat((cls_features, cnn_features), dim=1)

            x = self.relu(self.fc(concat_features))
            x = self.classifier(x)
            output = self.softmax(x)
            return output