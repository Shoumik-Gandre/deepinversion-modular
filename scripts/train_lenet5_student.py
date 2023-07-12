from dataclasses import dataclass
from pathlib import Path
import fire
import torch
import os
import sys

from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm 

sys.path.insert(0, os.path.abspath('..'))
from models.lenet_bn import LeNet5BN



@dataclass
class StudentTrainerHyperparams:
    epochs: int
    batch_size: int
    teacher_temperature: float
    optimizer: torch.optim.Optimizer


class StudentTrainer:

    def __init__(
            self,
            teacher: torch.nn.Module,
            student: torch.nn.Module,
            model_save_path: Path,
            train_dataset: Dataset,
            test_dataset: Dataset,
            hyperparams: StudentTrainerHyperparams,
            device = torch.device('cuda')
        ):
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.model_save_path = model_save_path

        self.train_dataloader = DataLoader(
            dataset=train_dataset, 
            batch_size=hyperparams.batch_size, 
            shuffle=True, 
            num_workers=0
        )
        self.eval_dataloader = DataLoader(
            dataset=test_dataset, 
            batch_size=hyperparams.batch_size, 
            num_workers=0
        )
        self.hyperparams = hyperparams
        self.criterion_train = torch.nn.KLDivLoss(reduction='batchmean')
        self.criterion_test = torch.nn.CrossEntropyLoss()
        self.device = device

    def train_step(self):
        self.student.train()
        self.teacher.eval()
        for i, (images, labels) in enumerate(pbar := tqdm(self.train_dataloader)):
            images = images.to(self.device) 
            labels = labels.to(self.device)
            teacher_output = F.softmax(self.teacher(images) / self.hyperparams.teacher_temperature, dim=1)
            student_output = F.log_softmax(self.student(images), dim=1)

            self.hyperparams.optimizer.zero_grad()
            loss = self.criterion_train(student_output, teacher_output.detach())
            loss.backward()
            self.hyperparams.optimizer.step()
            pbar.set_description(f'Loss {loss.item()}')

    def eval_step(self):
        self.student.eval()
        total_correct = 0
        avg_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.eval_dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = F.softmax(self.student(images), dim=1)

                avg_loss += self.criterion_test(outputs, labels)
                prediction = outputs.argmax(dim=1)
                total_correct += prediction.eq(labels.data.view_as(prediction)).sum()

        avg_loss /= len(self.eval_dataloader.dataset) # type: ignore
        acc = float(total_correct) / len(self.eval_dataloader.dataset) # type: ignore
        print('Test Avg. Loss: %f, Accuracy: %f' %
              (avg_loss.item(), acc)) # type: ignore

    def train(self):
        for epoch in range(1, self.hyperparams.epochs+1):
            print(f"Epoch [{epoch}/{self.hyperparams.epochs}]")
            self.train_step()
            self.eval_step()
        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.student, self.model_save_path)



def main(
        real_data_root: str,
        synthetic_data_root: str,
        teacher_model_path: str,
        model_save_path: str
    ):    

    device = torch.device('cuda')
    # Synthetic Dataset
    train_dataset = datasets.ImageFolder(
        root=synthetic_data_root, 
        transform=transforms.Compose([  
            # transforms.Grayscale(3),
            # transforms.Resize((32, 32)),
            transforms.ToTensor(),
            # KeepChannelsTransform((0,)),
            # transforms.Normalize((0.1307,), (0.3081,)),
        ])
    )
    _images = torch.stack([train_data[0] for train_data in train_dataset])
    _labels = torch.tensor([train_data[1] for train_data in train_dataset], dtype=torch.long)

    train_dataset = TensorDataset(_images, _labels)
    

    # Test Dataset
    eval_dataset = datasets.MNIST(
        root=real_data_root,
        train=False,
        transform=transforms.Compose([  
            transforms.Grayscale(3),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            # KeepChannelsTransform((0,)),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]),
        download=True
    )
    eval_images = torch.stack([train_data[0] for train_data in eval_dataset])
    eval_labels = torch.tensor([train_data[1] for train_data in eval_dataset], dtype=torch.long)
    eval_dataset = TensorDataset(eval_images, eval_labels)

    Path(teacher_model_path).parent.mkdir(parents=True, exist_ok=True)
    teacher: LeNet5BN = torch.load(teacher_model_path, device).eval()
    student = LeNet5BN(in_channels=3, num_labels=10).to(device).train()

    hyperparams = StudentTrainerHyperparams(
        epochs=100,
        batch_size=256,
        teacher_temperature=5,
        optimizer=torch.optim.Adam(student.parameters())
    )

    trainer = StudentTrainer(
        teacher=teacher,
        student=student,
        model_save_path=Path(model_save_path),
        train_dataset=train_dataset,
        test_dataset=eval_dataset,
        hyperparams=hyperparams,
        device=device
    )
    trainer.train()


if __name__ == '__main__':
    fire.Fire(main)
