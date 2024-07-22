#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy

import torch


# In[2]:


from models.latte_t2v import LatteT2V  # 导入模型
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
from diffusers.schedulers import PNDMScheduler
import torch
from einops import repeat
import copy

import torch

pretrain_model_path = '/data2/wangtuo/workspace/model/Latte/hf_hub/models--maxin-cn--Latte/t2v_required_models'
model_name = '/data2/wangtuo/workspace/model/Latte/hf_hub/models--maxin-cn--Latte/t2v.pt'
device = "cuda:1"

# T2v
lattet2v_model = LatteT2V.from_pretrained_2d(pretrain_model_path, subfolder="transformer", video_length = 16).to(device) 
from transformers import T5EncoderModel, T5Tokenizer
checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
lattet2v_model.load_state_dict(checkpoint['model'])
# 把pipeline加载明白

from sample.pipeline_videogen import VideoGenPipeline


vae = AutoencoderKL.from_pretrained(pretrain_model_path, subfolder="vae", torch_dtype=torch.float16).to(device)
vae.half()
print(vae.dtype)
tokenizer = T5Tokenizer.from_pretrained(pretrain_model_path, subfolder="tokenizer")
text_encoder = T5EncoderModel.from_pretrained(pretrain_model_path, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
scheduler = PNDMScheduler.from_pretrained(pretrain_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=0.0001, 
                                                  beta_end=0.02, 
                                                  beta_schedule="linear",
                                                variance_type="learned_range")
vae.eval()
text_encoder.eval()

videogen_pipeline = VideoGenPipeline(vae=vae, 
                                text_encoder=text_encoder, 
                                tokenizer=tokenizer, 
                                scheduler=scheduler, 
                                transformer=lattet2v_model).to(device)


# 一些参数配置
erase_scale = 0.1
preserve_scale =0.1
is_use_k = True
old_texts_ = ['Biden']
new_texts_ = ['Messi']
ret_texts = ['', 'A.J.Casson', 'Aaron Douglas', 'Aaron Horkey', 'Aaron Jasinski', 'Aaron Siskind', 'Abbott Fuller Graves', 'Abbott Handerson Thayer', 'Abdel Hadi Al Gazzar', 'Abed Abdi', 'Abigail Larson', 'Abraham Mintchine', 'Abraham Pether', 'Abram Efimovich Arkhipov', 'Adam Elsheimer', 'Adam Hughes', 'Adam Martinakis', 'Adam Paquette', 'Adi Granov', 'Adolf Hirémy-Hirschl', 'Adolph Gottlieb', 'Adolph Menzel', 'Adonna Khare', 'Adriaen van Ostade', 'Adriaen van Outrecht', 'Adrian Donoghue', 'Adrian Ghenie', 'Adrian Paul Allinson', 'Adrian Smith', 'Adrian Tomine', 'Adrianus Eversen', 'Afarin Sajedi', 'Affandi', 'Aggi Erguna', 'Agnes Cecile', 'Agnes Lawrence Pelton', 'Agnes Martin', 'Agostino Arrivabene', 'Agostino Tassi', 'Ai Weiwei', 'Ai Yazawa', 'Akihiko Yoshida', 'Akira Toriyama', 'Akos Major', 'Akseli Gallen-Kallela', 'Al Capp', 'Al Feldstein', 'Al Williamson', 'Alain Laboile', 'Alan Bean', 'Alan Davis', 'Alan Kenny', 'Alan Lee', 'Alan Moore', 'Alan Parry', 'Alan Schaller', 'Alayna Lemmer', 'Albert Benois', 'Albert Bierstadt', 'Albert Bloch', 'Albert Dubois-Pillet', 'Albert Eckhout', 'Albert Edelfelt', 'Albert Gleizes', 'Albert Goodwin', 'Albert Joseph Moore', 'Albert Koetsier', 'Albert Kotin', 'Albert Lynch', 'Albert Marquet', 'Albert Pinkham Ryder', 'Albert Robida', 'Albert Servaes', 'Albert Tucker', 'Albert Watson', 'Alberto Burri', 'Alberto Giacometti', 'Alberto Magnelli', 'Alberto Seveso', 'Alberto Sughi', 'Alberto Vargas', 'Albrecht Anker', 'Albrecht Durer', 'Alejandro Burdisio', 'Alejandro Jodorowsky', 'Aleksey Savrasov', 'Aleksi Briclot', 'Alena Aenami', 'Alessandro Allori', 'Alessandro Barbucci', 'Alessio Albi', 'Alex Andreev', 'Alex Colville', 'Alex Figini', 'Alex Garant', 'Alex Grey', 'Alex Gross', 'Alex Hirsch', 'Alex Horley', 'Alex Howitt', 'Alex Katz', 'Alex Maleev', 'Alex Petruk', 'Alex Prager', 'Alex Ross', 'Alex Russell Flint', 'Alex Schomburg', 'Alex Timmermans', 'Alex Toth', 'Alexander Archipenko', 'Alexander Bogen', 'Alexander Fedosav', 'Alexander Jansson', 'Alexander Kanoldt', 'Alexander McQueen', 'Alexander Millar', 'Alexander Milne Calder', 'Alexandr Averin', 'Alexandre Benois', 'Alexandre Cabanel', 'Alexandre Calame', 'Alexandre Jacovleff', 'Alexandre-Évariste Fragonard', 'Alexei Harlamoff', 'Alexej von Jawlensky', 'Alexey Kurbatov', 'Alexis Gritchenko', 'Alfred Augustus Glendening', 'Alfred Cheney Johnston', 'Alfred Eisenstaedt', 'Alfred Guillou', 'Alfred Heber Hutty', 'Alfred Henry Maurer', 'Alfred Kelsner', 'Alfred Kubin', 'Alfred Munnings', 'Alfred Parsons', 'Alfred Stevens', 'Alfredo Jaar', 'Algernon Blackwood', 'Alice Bailly', 'Alice Neel', 'Alice Pasquini', 'Alice Rahon', 'Alison Bechdel', 'Allen Williams', 'Allie Brosh', 'Allison Bechdel', 'Alma Thomas', 'Alois Arnegger', 'Alphonse Mucha', 'Alphonse Osbert', 'Alpo Jaakola', 'Alson Skinner Clark', 'Alvar Aalto', 'Alvaro Siza', 'Alvin Langdon Coburn', 'Alyssa Monks', 'Amadou Opa Bathily', 'Amanda Clark', 'Amandine Van Ray', 'Ambrosius Benson', 'Ambrosius Bosschaert', 'Amedee Ozenfant', 'Amedeo Modigliani', 'Amiet Cuno', 'Aminollah Rezaei', 'Amir Zand', 'Amy Earles', 'Amy Judd', 'Amy Sillman', 'Amédée Guillemin', 'Anato Finnstark', 'Anatoly Metlan', 'Anders Zorn', 'Ando Fuchs', 'Andre De Dienes', 'Andre Derain', 'Andre Kertesz', 'Andre Kohn', 'Andre-Charles Boulle', 'Andrea Kowch', 'Andrea Mantegna', 'Andreas Achenbach', 'Andreas Franke', 'Andreas Gursky', 'Andreas Rocha', 'Andreas Vesalius', 'Andrei Markin', 'Andrew Ferez', 'Andrew Macara', 'Andrew Robinson', 'Andrew Whem', 'Andrew Wyeth', 'Andrey Remnev', 'Android Jones', 'Andrzej Sykut', 'André Lhote', 'André Masson', 'Andy Fairhurst', 'Andy Goldsworthy', 'Andy Kehoe', 'Andy Warhol', 'Angela Barrett', 'Angela Sung', 'Angus McKie', 'Anish Kapoor', 'Anita Malfatti', 'Anja Millen', 'Anja Percival', 'Anka Zhuravleva', 'Ann Stookey', 'Anna Ancher', 'Anna Bocek', 'Anna Dittmann', 'Anna Razumovskaya', 'Anna and Elena Balbusso', 'Anne Brigman', 'Anne Dewailly', 'Anne Mccaffrey', 'Anne Packard', 'Anne Rothenstein', 'Anne Stokes', 'Anne Sudworth', 'Anne Truitt', 'Anne-Louis Girodet', 'Anni Albers', 'Annibale Carracci', 'Annick Bouvattier', 'Annie Soudain', 'Annie Swynnerton', 'Ansel Adams', 'Anselm Kiefer', 'Antanas Sutkus', 'Anthony Gerace', 'Anthony Thieme', 'Anthony van Dyck', 'Anto Carte', 'Antoine Blanchard', 'Antoine Verney-Carron', 'Anton Corbijn', 'Anton Domenico Gabbiani', 'Anton Fadeev', 'Anton Mauve', 'Anton Otto Fischer', 'Anton Pieck', 'Anton Raphael Mengs', 'Anton Semenov', 'Antonello da Messina', 'Antoni Gaudi', 'Antonio Canova', 'Antonio Donghi', 'Antonio J. Manzanedo', 'Antonio Mancini', 'Antonio Mora', 'Antonio Roybal', 'Antony Gormley', 'Apollinary Vasnetsov', 'Apollonia Saintclair', 'Aquirax Uno', 'Archibald Thorburn', 'Aries Moross', 'Arik Brauer', 'Aristarkh Lentulov', 'Aristide Maillol', 'Arkhyp Kuindzhi', 'Armand Guillaumin', 'Armand Point', 'Arnold Bocklin', 'Arnold Böcklin', 'Arnold Schoenberg', 'Aron Demetz', 'Aron Wiesenfeld', 'Arshile Gorky', 'Art Fitzpatrick', 'Art Frahm', 'Art Spiegelman', 'Artem Chebokha', 'Artemisia Gentileschi', 'Artgerm', 'Arthur Adams', 'Arthur Boyd', 'Arthur Dove', 'Arthur Garfield Dove', 'Arthur Hacker', 'Arthur Hughes', 'Arthur Lismer', 'Arthur Rackham', 'Arthur Radebaugh', 'Arthur Sarnoff', 'Arthur Streeton', 'Arthur Tress', 'Arthur Wardle', 'Artur Bordalo', 'Arturo Souto', 'Artus Scheiner', 'Ary Scheffer', 'Asaf Hanuka', 'Asger Jorn', 'Asher Brown Durand', 'Ashley Willerton', 'Atay Ghailan', 'Atelier Olschinsky', 'Atey Ghailan', 'Aubrey Beardsley', 'Audrey Kawasaki', 'August Friedrich Schenck', 'August Macke', 'August Sander', 'August von Pettenkofen', 'Auguste Herbin', 'Auguste Mambour', 'Auguste Toulmouche', 'Augustus Edwin Mulready', 'Augustus Jansson', 'Augustus John', 'Austin Osman Spare', 'Axel Törneman', 'Ayami Kojima', 'Ayan Nag', 'Aykut Aydogdu', 'Bakemono Zukushi', 'Balthus', 'Barbara Hepworth', 'Barbara Kruger', 'Barbara Stauffacher Solomon', 'Barbara Takenaga', 'Barclay Shaw', 'Barkley L. Hendricks', 'Barnett Newman', 'Barry McGee', 'Barry Windsor Smith', 'Bart Sears', 'Barthel Bruyn the Elder', 'Barthel Bruyn the Younger', 'Bartolome Esteban Murillo', 'Basil Gogos', 'Bastien Lecouffe-Deharme', 'Bayard Wu', 'Beauford Delaney', 'Beeple', 'Bella Kotak', 'Ben Aronson', 'Ben Goossens', 'Ben Hatke', 'Ben Nicholson', 'Ben Quilty', 'Ben Shahn', 'Ben Templesmith', 'Ben Wooten', 'Benedetto Caliari', 'Benedick Bana', 'Benoit B. Mandelbrot', 'Berend Strik', 'Bernard Aubertin', 'Bernard Buffet', 'Bernardo Bellotto', 'Bernardo Strozzi', 'Berndnaut Smilde', 'Bernie Wrightson', 'Bert Hardy', 'Bert Stern', 'Berthe Morisot', 'Bertil Nilsson', 'Bess Hamiti', 'Beth Conklin', 'Bettina Rheims', 'Bhupen Khakhar', 'Bill Brandt', 'Bill Carman', 'Bill Durgin', 'Bill Gekas', 'Bill Henson', 'Bill Jacklin', 'Bill Medcalf', 'Bill Sienkiewicz', 'Bill Viola', 'Bill Ward', 'Bill Watterson', 'Billy Childish', 'Bjarke Ingels', 'Blek Le Rat', 'Bo Bartlett', 'Bo Chen', 'Bob Byerley', 'Bob Eggleton', 'Bob Ross', 'Bojan Jevtic', 'Bojan Koturanovic', 'Bordalo II', 'Boris Grigoriev', 'Boris Groh', 'Boris Kustodiev', 'Boris Vallejo', 'Botero', 'Brad Kunkle', 'Brad Rigney', 'Brandon Mably', 'Brandon Woelfel', 'Brenda Zlamany', 'Brent Cotton', 'Brent Heighton', 'Brett Weston', 'Brett Whiteley', 'Brian Bolland', 'Brian Despain', 'Brian Froud', 'Brian K. Vaughan', 'Brian Kesinger', 'Brian Mashburn', 'Brian Oldham', 'Brian Stelfreeze', 'Brian Sum', 'Briana Mora', 'Brice Marden', 'Bridget Bate Tichenor', 'Briton Rivière', 'Brooke DiDonato', 'Brooke Shaden', 'Brothers Grimm', 'Brothers Hildebrandt', 'Bruce Munro', 'Bruce Nauman', 'Bruce Pennington', 'Bruce Timm', 'Bruno Catalano', 'Bruno Munari', 'Bruno Walpoth', 'Bryan Hitch', 'Butcher Billy', 'C. R. W. Nevinson', 'Cagnaccio Di San Pietro', 'Camille Corot', 'Camille Pissarro', 'Camille Walala', 'Canaletto', 'Candido Portinari', 'Carel Willink', 'Carl Barks', 'Carl Gustav Carus', 'Carl Holsoe', 'Carl Larsson', 'Carl Spitzweg', 'Carlo Crivelli', 'Carlos Schwabe', 'Carmen Saldana', 'Carne Griffiths', 'Casey Weldon', 'Caspar David Friedrich', 'Cassius Marcellus Coolidge', 'Catrin Welz-Stein', 'Cedric Peyravernay', 'Chad Knight', 'Chantal Joffe', 'Charles Addams', 'Charles Angrand', 'Charles Blackman', 'Charles Camoin', 'Charles Dana Gibson', 'Charles E. Burchfield', 'Charles Gwathmey', 'Charles Le Brun', 'Charles Liu', 'Charles Schridde', 'Charles Schulz', 'Charles Spencelayh', 'Charles Vess', 'Charles-Francois Daubigny', 'Charlie Bowater', 'Charline von Heyl', 'Chaïm Soutine', 'Chen Zhen', 'Chesley Bonestell', 'Chiharu Shiota', 'Ching Yeh', 'Chip Zdarsky', 'Chris Claremont', 'Chris Cunningham', 'Chris Foss', 'Chris Leib', 'Chris Moore', 'Chris Ofili', 'Chris Saunders', 'Chris Turnham', 'Chris Uminga', 'Chris Van Allsburg', 'Chris Ware', 'Christian Dimitrov', 'Christian Grajewski', 'Christophe Vacher', 'Christopher Balaskas', 'Christopher Jin Baron', 'Chuck Close', 'Cicely Mary Barker', 'Cindy Sherman', 'Clara Miller Burd', 'Clara Peeters', 'Clarence Holbrook Carter', 'Claude Cahun', 'Claude Monet', 'Clemens Ascher', 'Cliff Chiang', 'Clive Madgwick', 'Clovis Trouille', 'Clyde Caldwell', 'Coby Whitmore', 'Coles Phillips', 'Colin Geller', 'Conor Harrington', 'Conrad Roset', 'Constant Permeke', 'Constantin Brancusi', 'Cory Arcangel', 'Cory Loftis', 'Costa Dvorezky', 'Craig Davison', 'Craig Mullins', 'Craig Wylie', 'Craola', 'Cyril Rolando', 'Dain Yoon', 'Damien Hirst', 'Dan Flavin', 'Dan Mumford', 'Dan Witz', 'Daniel Buren', 'Daniel Clowes', 'Daniel Garber', 'Daniel Merriam', 'Daniel Ridgway Knight', 'Daniela Uhlig', 'Daniele Afferni', 'Dante Gabriel Rossetti', 'Dao Le Trong', 'Dariusz Zawadzki', 'Darren Bacon', 'Darwyn Cooke', 'Daryl Mandryk', 'Dave Dorman', 'Dave Gibbons', 'Dave McKean', 'David A. Hardy', 'David Aja', 'David B. Mattingly', 'David Bomberg', 'David Bowie', 'David Burdeny', 'David Burliuk', 'David Driskell', 'David Finch', 'David Hockney', 'David Inshaw', 'David Ligare', 'David Lynch', 'David McClellan', 'David Palumbo', 'David Shrigley', 'David Spriggs', 'David Teniers the Younger', 'David Wiesner', 'Dean Cornwell', 'Dean Ellis', 'Death Burger', 'Debbie Criswell', 'Derek Boshier', 'Diane Arbus', 'Diane Dillon', 'Dick Bickenbach', 'Diego Dayer', 'Diego Rivera', 'Diego Velázquez']


# In[6]:


def get_prompt_embedding(videogen_pipeline, prompt):
    encoder_hidden_state, encoder_hidden_state_mask = videogen_pipeline.get_sentence_embedding(prompt, 
                        video_length=16, 
                        height=512, 
                        width=512, 
                        num_inference_steps=4,
                        guidance_scale=7.5,
                        enable_temporal_attentions=True,
                        num_images_per_prompt=1,
                        mask_feature=True,
                        enable_vae_temporal_decoder=False
                        )
    return encoder_hidden_state[16,:,:]


# In[7]:


transformers_block = lattet2v_model.transformer_blocks.named_children()
ca_layers = []
print(f'寻找交叉注意力')
for transformer_block in transformers_block: 
    # 对于每一层transformer_block,选择basic_block
    basic_transformer_block = transformer_block[1]
    basic_transformer_layers = basic_transformer_block.named_children()
    # 遍历，找到基本层
    for layer in basic_transformer_layers:
        if 'attn2' in layer[0]:
            ca_layers.append(layer[1])
            
print(ca_layers[0].to_v.weight)  # 28个交叉注意力层

# 这里的encoder_hidden_states还是作为k,v的输入
projection_matrices = [l.to_v for l in ca_layers]
og_matrices = [copy.deepcopy(l.to_v) for l in ca_layers]
if is_use_k:
    projection_matrices = projection_matrices + [l.to_k for l in ca_layers]
    og_matrices = og_matrices + [copy.deepcopy(l.to_k) for l in ca_layers]
print(projection_matrices)

# 这里面很重要的一部分，ca_layers就是attention
# 重置参数,这部分可以修改一下
num_caption_layer = len(ca_layers)
for idx, layer in enumerate(ca_layers):
    layer.to_v = copy.deepcopy(projection_matrices[idx])
    projection_matrices[idx] = layer.to_v
    if is_use_k:
        layer.to_k = copy.deepcopy(og_matrices[num_caption_layer+idx])
        projection_matrices[num_caption_layer+idx] = layer.to_k
        
old_texts = []
new_texts = []
for old_text, new_text in zip(old_texts_, new_texts_):
    old_texts.append(old_text)
    n_t = new_text
    if n_t == '':
        n_t = ' '
    new_texts.append(n_t)

lamb = 0.01


# In[8]:


# 重新写一个概念映射
for layer_num in range(len(projection_matrices)):
    with torch.no_grad():
        position1 = torch.zeros((1152, 1152)).to(device)
        position2 = torch.zeros((1152, 1152)).to(device)
        position3 = torch.zeros((1152, 1152)).to(device)
        position4 = torch.zeros((1152, 1152)).to(device)

        for old_text, new_text in zip(old_texts, new_texts):
            # 获取missi c^T
            new_embeds_cap_spatial = get_prompt_embedding(videogen_pipeline, new_text)  # [3,1152]
            # vi
            old_embeds_cap_spatial = get_prompt_embedding(videogen_pipeline, old_text)
            layer = projection_matrices[layer_num]
            vi = layer(new_embeds_cap_spatial).detach()  # [3,1152]
            
            position1 += old_embeds_cap_spatial.transpose(0, 1) @ vi
            
            
            position3 += old_embeds_cap_spatial.transpose(0, 1) @ old_embeds_cap_spatial
        
        for remain_text in ret_texts:
            remain_embeds_cap_spatial = get_prompt_embedding(videogen_pipeline, remain_text)
            layer = projection_matrices[layer_num]
            vi = layer(remain_embeds_cap_spatial).detach()
            position2 += remain_embeds_cap_spatial.transpose(0, 1) @ vi
            
            position4 += remain_embeds_cap_spatial.transpose(0, 1) @ remain_embeds_cap_spatial
            
        mat1 = position1 + position2
        mat2 = position3 + position4
        
        
        projection_matrices[layer_num].weight = torch.nn.Parameter(mat1 @ torch.inverse(mat2))
        if layer_num == 0:
            print(mat1)
            print(mat2)
            print(projection_matrices[layer_num].weight)
checkpoint = { "model": lattet2v_model.state_dict() }
torch.save(checkpoint, 'lattet2v_model2.pt')
        


# In[15]:




# In[9]:


# for layer_num in range(len(projection_matrices)):
#     with torch.no_grad():
#         # 不改变梯度, 这是后面推导的公式
#         mat1 = lamb * projection_matrices[layer_num].weight
#         mat2 = lamb * torch.eye(projection_matrices[layer_num].weight.shape[1], device = projection_matrices[layer_num].weight.device)
#         for idx, t in enumerate(zip(old_texts, new_texts)):
#             print(f't{t}')
#             old_text, new_text = t
#             old_embeds_cap_spatial = get_prompt_embedding(videogen_pipeline, old_text)
#             new_embeds_cap_spatial = get_prompt_embedding(videogen_pipeline, new_text)
#             print(f'old_embeds_cap_spatial{old_embeds_cap_spatial.shape},new_embeds_cap_spatial{new_embeds_cap_spatial.shape}')
#             context = old_embeds_cap_spatial.detach()
#             # print(f'context{context.shape}')
#             values = []
            
#             with torch.no_grad():
#                 for layer in projection_matrices:
#                     o_embs = layer(old_embeds_cap_spatial).detach()  # v*
#                     u = o_embs
#                     u = u / u.norm()
                    
#                     new_embs = layer(new_embeds_cap_spatial).detach()
#                     new_embs_proj = (u*new_embs).sum()
                    
#                     traget = new_embs - (new_embs_proj * u)
#                     values.append(traget.detach())
                    
            
#             context_vector = context # c [token,1152]
#             context_vector_T = context.transpose(0, 1)  # [1152, token]
            
#             value_vector = values[layer_num]
#             for_mat1 = (context_vector_T @ value_vector)  # [1152,1152]
#             for_mat2 = (context_vector_T @ context_vector)  # [1152,1152]
#             # print(f'format1{for_mat1.shape},format2{for_mat2.shape},mat1{mat1.shape}')
#             mat1 += erase_scale*for_mat1
#             mat2 += erase_scale*for_mat2
            
#         # # 对于其他要保存的参数，我们要保存下来
#         # for old_text, new_text in zip(ret_texts, ret_texts):
#         #     print(f'old_text{old_text},new_text{new_text}')
#         #     old_embeds_cap_spatial = get_prompt_embedding(videogen_pipeline, old_text)
#         #     new_embeds_cap_spatial = get_prompt_embedding(videogen_pipeline, new_text)
            
            
#         #     context = old_embeds_cap_spatial.detach()
#         #     values = []
            
#         #     with torch.no_grad():
#         #         for layer in projection_matrices:
#         #             values.append(layer(new_embeds_cap_spatial.detach()))
#         #     context_vector = context # c [32,120,1152]
#         #     context_vector_T = context.transpose(0, 1)  # cT  [32,1152,120] []
            
#         #     value_vector = values[layer_num]
#         #     for_mat1 = (context_vector_T @ value_vector)  # [1152,1152]
#         #     for_mat2 = (context_vector_T @ context_vector)
#         #     mat1 += preserve_scale*for_mat1
#         #     mat2 += preserve_scale*for_mat2
        
#         projection_matrices[layer_num].weight = torch.nn.Parameter((mat1 @ torch.inverse(mat2)))
        
# checkpoint = { "model": lattet2v_model.state_dict() }
# torch.save(checkpoint, 'lattet2v_model2.pt')


# In[10]:


# 这里面我要确认是否真的改变参数
l_to_v_old = ca_layers[0].to_v.weight
print(l_to_v_old)
l_to_v_new = projection_matrices[0].weight
print(l_to_v_new)
og_to_v = og_matrices[0].weight
print(og_to_v)

import numpy as np


ans = mat2.cpu().numpy()
ans
def is_full_rank_matrix_C(A):
    # 计算矩阵的行阶梯形式
    rref = np.linalg.matrix_rank(A)
    return rref == A.shape[0]

print(is_full_rank_matrix_C(ans))
