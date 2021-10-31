function re = collect_data(N, M, res)

N_M_matrix = [];
row_out = [];

for i=1:N
    for j=1:M
        s =struct;
        bathy_file = fopen(join(['maps/bathy_', int2str(i)]));
        %random_details = fopen('data');
        shy_output_x = join(['SHY_OUTPUT/matrix_vel_x_', int2str(i),'.mat']);
        shy_output_y = join(['SHY_OUTPUT/matrix_vel_y_', int2str(i),'.mat']);

        ssp_file = fopen(join(['INPUT/ssp_',int2str(i)]));
        
        bellhop_output = join(['OUTPUT/ir_', int2str(i), int2str(j),'.mat']);
        drift_output = join(['OUTPUT/location_output_', int2str(i), int2str(j),'.csv']);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Save the bathy
        a = fscanf(bathy_file,'%f');
        bathy = [];
        row = [];
        for k=1:size(a)
            if k > 7
                if size(row, 2) == res
                    bathy = [bathy;row];
                    row = [];
                end
                if size(row, 2) < res
                    row = [row,a(k)];
                end
            end
        end
        bathy = [bathy;row];
        s.bathy = bathy;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Save the SHYFEM
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% output
        %shy_output_x = fopen('C:\technion\new\SHY_OUTPUT_mat\matrix_vel_x.mat')
        
        s.shy_x = load(shy_output_x);
        s.shy_y = load(shy_output_y);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Save the ssp
        a = fscanf(ssp_file,'%f');
        ssp = [];
        row = [];
        for k=1:size(a)
            row = [row,a(k)];
            if mod(k,2) == 0
                ssp = [ssp;row];
                row = [];
            end
        end
        s.ssp = ssp;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Save location_output
        s.location = readtable(drift_output);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Save ir_output
        s.ir = load(bellhop_output);
        
        row_out = [row_out, s];
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Close files
        
        fclose(bathy_file);
        fclose(ssp_file);
    end
    q = i
    N_M_matrix = [N_M_matrix; row_out];
    row_out = [];
end
finish = 'finish'

save 'a.mat' N_M_matrix
                
