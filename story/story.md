Welcome to the story behind this project.

故事从2021年开始，当时我在入门一些显微光场方面的研究，那么自然绕不开的一关是点扩散函数。当然如果不求甚解的话，用别人预先计算好的的光场点扩散函数也不妨碍应用在比如前向投影和三维重建中。作为一个初学者，我还是选择一点点搞明白光场点扩散函数的原理——这对于一个非光学专业的我来说并不容易，我几乎花了整整三个月才搞明白这些各种细节，当时我是参考：
[doi.org/10.1038/nmeth.2964](https://doi.org/10.1038/nmeth.2964)
[doi.org/10.1016/j.cell.2021.04.029](https://doi.org/10.1016/j.cell.2021.04.029)

这两篇文章和对应的代码。最大的印象就是，这玩意非常慢，哪怕有了nvidia显卡的并行加持，基本上也需要小时级别的时间。
插入：注意这两个的psf并不一样，第一个就是经典的空间非均匀的点扩散函数，物方除了各个深度之外，还有微透镜共轭对应的Nnum x Nnum种不同的可能性，也就是原始光场点扩散函数是Nnum x Nnum x depth x psfH x psfW的大小，这样的成像时物方不同位置的点要对应不同的点扩散函数，叠加得到光场的宏像素图(MacPI, Macro-Pixel Image)，也是像面微透镜对应的光场传感器采集到的图。
我们都知道光场采集牺牲了空间采样率换取了角度采样率，那么如果不牺牲空间采样率，光场应该是什么样子？这个就是扫描光场，也就是第二个文章引入的概念，它把一般光场认为是扫描光场的一个空间抽样。核心思想是，对于任一深度，既然物方不同位置可能对应Nnum x Nnum个不同的成像逻辑，那么让物体空间横向平移Nnum x Nnum次，就可以遍历出全部的互异的可能性，把这个结果按横向平移的相对位置拼起来，就得到了不牺牲空间采样率的光场，也就是全扫描的光场。实际硬件系统是通过傅里叶面galvo的旋转实现平移，多次曝光加上空间重排得到实采的扫描光场，至于psf就直接从子孔径图建模(SAI, Sub-Aperture Image)，也就是相空间一致的(任一角度空间一致的)点扩散函数。

长期以来我们被这个极其缓慢的过程困扰着，也自然限制了很多探索的方向。核心在于，这个点扩散函数的计算是在宽场点扩散函数的基础上，Depth x Nnum x Nnum次的波前穿过微透镜阵列，再加上菲涅尔衍射得到，这个过程空间和角度还是互相耦合的，因此我们依次做了如下尝试：
1、把原来的debye积分换成角谱传播，这样宽场那部分的计算效率可以提高很多。
https://doi.org/10.1038/s41467-021-26730-w
2、GPU，并行化，向量化
GPU并行化这个好理解，向量化就是把一些for循环计算的变量，用内存空间连续的数组来存储和操作，主要表现在psf的空间重排映射：

```
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% %%%%%%%%%%%%%%%% Convert to phase-space PSF  %%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% IMGsize=size(H,1)-mod((size(H,1)-Nnum),2*Nnum);
% psf=zeros(IMGsize,IMGsize,Nnum,Nnum,size(H,5)); %% phase-space PSF
% for z=1:size(H,5)
 
%     sLF=zeros(IMGsize,IMGsize,Nnum,Nnum); %% scanning light field images
%     index1=round(size(H,1)/2)-fix(size(sLF,1)/2);
%     index2=round(size(H,1)/2)+fix(size(sLF,1)/2);
%     for ii=1:size(H,3)
%         for jj=1:size(H,4)
%             sLF(:,:,ii,jj)=im_shift3(squeeze(H(index1:index2,index1:index2,ii,jj,z)),ii-((Nnum+1)/2), jj-(Nnum+1)/2);
%         end
%     end
  
%     multiWDF=zeros(Nnum,Nnum,size(sLF,1)/size(H,3),size(sLF,2)/size(H,4),Nnum,Nnum); %% multiplexed phase-space
%     for i=1:size(H,3)
%         for j=1:size(H,4)
%             for a=1:size(sLF,1)/size(H,3)
%                 for b=1:size(sLF,2)/size(H,4)
%                     multiWDF(i,j,a,b,:,:)=squeeze(  sLF(  (a-1)*Nnum+i,(b-1)*Nnum+j,:,:  )  );
%                 end
%             end
%         end
%     end
%     WDF=zeros(  size(sLF,1),size(sLF,2),Nnum,Nnum  ); %% multiplexed phase-space
%     for a=1:size(sLF,1)/size(H,3)
%         for c=1:Nnum
%             x=Nnum*a+1-c;
%             for b=1:size(sLF,2)/size(H,4)
%                 for d=1:Nnum
%                     y=Nnum*b+1-d;
%                     WDF(x,y,:,:)=squeeze(multiWDF(:,:,a,b,c,d));
%                 end
%             end
%         end
%     end
%     psf(:,:,:,:,z)=WDF;
  
% end
% psf=single(psf);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%% Convert to phase-space PSF  %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
IMGsize=size(H,1)-mod((size(H,1)-Nnum),2*Nnum);
toc
psf= zeros(IMGsize,IMGsize,Nnum,Nnum,size(H,5),'single'); %% phase-space PSF

index1=round(size(H,1)/2)-fix(size(psf,1)/2);
index2=round(size(H,1)/2)+fix(size(psf,1)/2);
for ii=1:size(H,3)
    for jj=1:size(H,4)
        % psf(:,:,ii,jj,:)=im_shift2(squeeze(H(index1:index2,index1:index2,ii,jj,:)),ii-((Nnum+1)/2), jj-(Nnum+1)/2);
        % psf(:,:,ii,jj,:)=im_shift2(squeeze(H(index1:index2,index1:index2,ii,jj,:)),0, 0);
        psf(:,:,ii,jj,:)=squeeze(H(index1:index2,index1:index2,ii,jj,:));
    end
end

psf = reshape(psf,[Nnum,size(psf,1)/size(H,3),Nnum,size(psf,2)/size(H,4),Nnum,Nnum, size(H,5)]);
psf = permute(psf,[5,2,6,4,1,3,7]);
psf = flip(psf,1);
psf = flip(psf,3);
% psf = flip(psf,2);
% psf = flip(psf,4);
psf = reshape(psf,[IMGsize,IMGsize,Nnum,Nnum,size(H,5)]);
psf=single(psf);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
```

向量化就是把上边那些for循环，换成下边这些permute reshape操作，中间要注意配合一些imshift操作，具体可以参考main_computePSF_vdf_defocus.m这个文件。2021年第一次写完这一块加速还是很兴奋的，毕竟单就这一块而言确实提速了10倍以上，而且看起来确实更容易懂一些。

做完这些之后，可能再加上各种别的优化，对于一些大型psf（比如paper里的parameter2）还是很慢，大概算一个psf在6小时左右。

但对于实际应用来说，还是有些杯水车薪。关键就是两点，第一psf仿真过程它很慢，不管是标定系统，还是做数据集，都极其不方便。其次psf很大，一个psf往往在10 GB大小以上(15x15x101x375x375)，在某些大视差的系统可能要更多(15x15x101x765x765)，从硬盘中加载到内存，或者放在gpu上都是一个很大的负担，这也打消了提前生成一个包含很多psf的数据集的思路——哪怕真的用很多资源算出来了，存在哪里，怎么读到内存里都是问题。

在无数次盯着psf时，我总在思考一个问题，这个psf形状其实并不复杂，就是一个shift加上一个特定形状的blur，但是怎么精确知道这两个东西？我曾经试过用神经网络拟合这个psf，虽然对某个psf总可以调得很好，但是效果总是不够满意。physical psf的连续渐变，以及如何保证来一套新的参数，这个模型还能work，都是问题。

终于有一天，在看代码时，我做了一个抽象：

```
a = 1:9;
a = padarray(a, [0,3], 0, 'both');
fresnel = [1:5,4,3,2,1];
fresnel = padarray(fresnel,[0,3], 0, 'both');
mla = [1,2,1,1,2,1,1,2,1];
mla = padarray(mla,[0,3], 0, 'circular');
res1 = zeros(3,9);
for ii = 1:3
    a_ = circshift(a,2-ii,2);
    temp = convn(a_ .* mla, fresnel,'same');
    res1(ii,:) = temp(:,1+3:end-3);
end
res1 = reshape(res1, 3, 3, 3);
res1 = permute(res1, [2,1,3]);
res1 = reshape(res1, [3,9]);

res2 = zeros(3,9);

for ii = 1:3
    fresnel_ = circshift(fresnel,2-ii,2);
    temp = convn(a , mla .* fresnel_,'same');
    res2(ii,:) = temp(:,1+3:end-3);
end

res11 = zeros(9,9);
for ii = 1:9
    a_ = circshift(a,5-ii,2);
    temp = convn(a_ .* mla, fresnel,'same');
    res11(ii,:) = temp(:,1+3:end-3);
end
res11 = res11(:,4:6)';



res3 = zeros(3,9);

for ii = 1:3
    mla_ = circshift(mla,ii-2,2);
    temp = convn(a , mla_ .* fresnel,'same');
    res3(ii,:) = temp(:,1+3:end-3);
end
```

上边就是这个核心，为什么res2可以和res1的结果一样？现在看来，这个简单的程序就是shift-fresnel模型的雏形。理论上它的计算量并没有改变，但是却把角度给解耦了，这样我想知道某些视角的psf，就不必计算全部的视角了。实际上，解耦已经帮了很大忙，因为解耦我们每次可以算更少的角度，显存压力更小，就更方便缓存和调度。另一方面，实际使用中我们发现很多系统边缘视角往往信噪比不高，就采用中心圆视角的方案，比如对于15x15我们就用中间的81个视角，从这种程度上说解耦计算确实更方便。

而解耦之后，分而治之逐个击破的想法就顺利成章，也就是shift-fresnel分解后的shift-blur，把计算更集中到有必要的计算中。

2026年很多人工智能模型能力突飞猛进，在人和AI的不断优化下，通过pytorch各种并行化和缓存的支持，非解耦版本的速度压缩到了36秒，也是感慨AI确实要改变世界了。比如给定一个场景，像差未知，需要通过优化的手段来确定更准确的psf时，一次计算36秒，那1000次迭代优化就是10个小时，在一些specimen-induced 变化的环境下还是没法用。好在我们现在突破了shift-fresnel with shift-blur 的超级方案，这个全部的过程终于可以在几分钟内搞定了。

也许可以更快，如果AI看到了这个shift-fresnel with shift-blur的方案，大概率能进一步实现更夸张的提速，不过我已经对它的progress相对满意了，点到为止。我相信有了这个工具，物理场景信息的联合观测估计可以更方便，相关科学问题的方法设计可以更有物理理论指导。
