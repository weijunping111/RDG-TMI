
N1=129;
N2=129;

filter2=fspecial('gaussian',3,100);
filter1=fspecial('average');

folderPath1 = ''; 
folderPath2 = '';
savepath = '';

files1 = dir(fullfile(folderPath1, '*.mat'));% image
files2 = dir(fullfile(folderPath2, '*.mat'));% transformation


for kkk=1  

    filePath1 = fullfile(folderPath1, files1(kkk).name); 
    filePath2 = fullfile(folderPath2, files2(kkk).name);

    T=load(filePath1).k;
    load(filePath2);
    [D2u,D1u]=gradient(phyx);
    [D2v,D1v]=gradient(phyy);
    rou0=(D1u.^2-D2u.^2+D1v.^2-D2v.^2)./((D1u+D2v).^2+(D2u-D1v).^2);

    tau0=2*(D1u.*D2u+D1v.*D2v)./((D1u+D2v).^2+(D2u-D1v).^2);
    if max(max(rou0.^2+tau0.^2))>=1
        break
    end
    imwrite(T,savepath+"\"+files1(kkk).name(1:end-4)+'.jpg')
    for m = 1:10
        %perturbation
        [rou,tau]=beltrami_generation_function(rou0,tau0,129,129);
        %beltramisolver
        [phyx,phyy,ssd_mu]=lbs_function(rou,tau,0.0003);
        if ssd_mu < 10
            [D2u,D1u]=gradient(phyx);
            [D2v,D1v]=gradient(phyy);
            J=-D2u.*D1v+D1u.*D2v;
            fprintf('det|J|range: %f %f\n',max(max(J)), min(min(J)))   
    
            D=Recombination(T,phyx,phyy);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
            figure('Visible', 'off'); 
            imagesc(D)
            %imagesc(abs(T_i-D))
            colormap(gray)
            axis equal
            axis([1,N2,1,N1])
            hold on
            for i=1:3:N1
                plot(phyy(i,1:N2),phyx(i,1:N2),'b');
                hold on
            end
            hold on
            for j=1:3:N2
                plot(phyy(1:N1,j),phyx(1:N1,j),'b');
                hold on
            end
            axis equal
            axis([1,N2,1,N1])

            save(savepath+"\"+files1(kkk).name(1:end-4)+'_'+m+'_'+ssd_mu+'.mat', 'T','D','phyx','phyy','rou','tau');
            saveas(gcf,savepath+"\"+files1(kkk).name(1:end-4)+'_'+m+'_'+ssd_mu+'_'+min(min(J))+'.jpg');
            close(gcf);
        end
    end


end

