Welcome to the story behind this project.

故事从 2021 年开始。当时我在入门一些显微光场方面的研究，自然绕不开的一关就是点扩散函数（PSF）。当然，如果不求甚解，用别人预先计算好的光场 PSF，也不妨碍把它应用在前向投影和三维重建中。

作为一个初学者，我还是选择一点点搞明白光场 PSF 的原理。这对于非光学专业的我来说并不容易：我几乎花了整整三个月，才搞明白其中的各种细节。当时参考的是下面这两篇文章和对应的代码：

- [doi.org/10.1038/nmeth.2964](https://doi.org/10.1038/nmeth.2964)
- [doi.org/10.1016/j.cell.2021.04.029](https://doi.org/10.1016/j.cell.2021.04.029)

最大的印象就是：这玩意非常慢。哪怕有了 NVIDIA 显卡的并行加持，基本上也需要小时级别的时间。

这里先插一句，这两篇文章里的 PSF 并不一样。

第一篇使用的是经典的空间非均匀点扩散函数。物方除了各个深度之外，还有微透镜共轭对应的 `Nnum × Nnum` 种不同可能，也就是原始光场 PSF 的大小为 `Nnum × Nnum × depth × psfH × psfW`。成像时，物方不同位置的点对应不同的 PSF，叠加得到光场的宏像素图（MacPI，Macro-Pixel Image），也就是像面微透镜对应的光场传感器采集到的图。

光场采集牺牲了空间采样率，换取角度采样率。那么，如果不牺牲空间采样率，光场应该是什么样子？这就是第二篇文章引入的扫描光场：它把一般光场看作扫描光场的一个空间抽样。

核心思想是：对于任一深度，既然物方不同位置可能对应 `Nnum × Nnum` 种不同的成像逻辑，那么让物体在空间中横向平移 `Nnum × Nnum` 次，就可以遍历全部互异的可能。再把结果按横向平移的相对位置拼起来，就得到了不牺牲空间采样率的光场，也就是全扫描光场。

实际硬件系统通过傅里叶面 galvo 的旋转实现平移，经过多次曝光和空间重排，得到实采的扫描光场。至于 PSF，则直接从子孔径图（SAI，Sub-Aperture Image）建模，也就是相空间一致、任一角度下空间一致的点扩散函数。

长期以来，我们一直被这个极其缓慢的计算过程困扰着，它也自然限制了很多探索的方向。核心在于：光场 PSF 是在宽场 PSF 的基础上，经过 `depth × Nnum × Nnum` 次波前穿过微透镜阵列，再加上菲涅尔衍射得到的。在这个计算过程中，空间和角度还是互相耦合的。因此，我们依次做了下面这些尝试。

首先，把原来的 Debye 积分换成角谱传播，这样可以大幅提高宽场部分的计算效率。相关参考：

[doi.org/10.1038/s41467-021-26730-w](https://doi.org/10.1038/s41467-021-26730-w)

接着是 GPU、并行化和向量化。GPU 并行化比较好理解；向量化，就是把一些原本在 `for` 循环中计算的变量，用内存空间连续的数组来存储和操作。这里主要体现在 PSF 的空间重排映射上：

<details>
<summary>展开相空间重排的代码对照：for 循环与 reshape / permute</summary>

```matlab
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

</details>

向量化就是把上面的那些 `for` 循环，换成下面这些 `permute`、`reshape` 操作，中间需要注意配合一些 `imshift` 操作，具体可以参考 [main_computePSF_vdf_defocus.m](psfcalc_ref/main_computePSF_vdf_defocus.m)。2021 年第一次写完这一块加速时，我还是很兴奋的：毕竟单就这一块而言，确实提速了十倍以上，而且看起来也更容易懂一些。

做完这些，再加上其他一些优化，对于大型 PSF，比如 paper 里的 Parameter 2，计算还是很慢，大概算一个 PSF 要六个小时左右。

对于实际应用来说，这样的改善还是有些杯水车薪。关键有两点。

一是 PSF 仿真过程很慢。不管是标定系统，还是做数据集，都极其不方便。

二是 PSF 很大。一个 PSF 往往在 10 GB 以上，比如 `15 × 15 × 101 × 375 × 375`；在一些大视差系统中，可能还要更大，比如 `15 × 15 × 101 × 765 × 765`。从硬盘加载到内存，或者放到 GPU 上，都是很大的负担。这也打消了提前生成一个包含很多 PSF 的数据集的思路——哪怕真的投入很多资源把它们算出来了，存在哪里、怎么读到内存里，也都是问题。

在无数次盯着 PSF 时，我总在思考：这个 PSF 的形状其实并不复杂，就是一个 shift 加上一个特定形状的 blur，但怎么才能精确知道这两个东西？

我曾经试过用神经网络拟合 PSF。虽然对某个 PSF 总可以调得很好，但效果始终不够满意。物理 PSF 的连续渐变，以及换一套新的参数之后，如何保证模型还能工作，都是问题。

终于有一天，在看代码时，我做了一个抽象：

```matlab
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

上面就是这个核心：为什么 `res2` 可以和 `res1` 的结果一样？现在看来，这个简单的程序，就是 Shift–Fresnel 模型的雏形。

理论上，它的计算量并没有改变，但是角度被解耦了。这样，想知道某些视角的 PSF，就不必计算全部视角。实际上，解耦已经帮了很大的忙：每次可以算更少的角度，显存压力更小，也就更方便缓存和调度。

另一方面，实际使用中我们发现，很多系统的边缘视角往往信噪比不高，因此会采用中心圆视角的方案。比如对于 `15 × 15` 个视角，我们就使用中间的 81 个视角。从这个角度看，解耦计算确实更方便。

而解耦之后，分而治之、逐个击破的想法就顺理成章了，也就是在 Shift–Fresnel 分解后进一步做 Shift–blur，把计算更集中到有必要的部分。

2026 年，很多人工智能模型的能力突飞猛进。在人和 AI 的不断优化下，通过 PyTorch 各种并行化和缓存的支持，非解耦版本的计算时间压缩到了 36 秒。也是感慨，AI 确实要改变世界了。

比如给定一个场景，像差未知，需要通过优化来确定更准确的 PSF。如果一次计算需要 36 秒，那么 1000 次迭代优化就是十个小时。在一些 specimen-induced 变化的环境下，还是没法用。好在我们现在突破了 Shift–Fresnel with Shift–blur 的超级方案，这整个过程终于可以在几分钟内搞定了。

也许还可以更快。如果 AI 看到了这个 Shift–Fresnel with Shift–blur 的方案，大概率能进一步实现更夸张的提速。不过，我已经对目前的 progress 相对满意了，点到为止。

我相信，有了这个工具，物理场景信息的联合观测估计可以更方便，相关科学问题的方法设计，也可以更有物理理论指导。
