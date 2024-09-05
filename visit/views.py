from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect, get_object_or_404
from django.views.generic import CreateView, ListView, DetailView, UpdateView, DeleteView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponse
from .forms import *
from django.shortcuts import render, redirect
from django.contrib import messages
from .decorators import *
from django.utils.decorators import method_decorator
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import io
import urllib, base64


# Create your views here.
def index(request):
    return render(request, 'visit/main.html', {'title': 'Metrics UP'})


def video(request):
    return render(request, 'visit/video.html', {'title': 'Metrics UP'})


def faq(request):
    return render(request, 'visit/faq.html', {'title': 'Metrics UP'})


def blog(request):
    return render(request, 'visit/blog.html', {'title': 'Metrics UP'})


def opinions(request):
    return render(request, 'visit/opinions.html', {'title': 'Metrics UP'})


# def login(request):
#     return render(request, 'visit/login.html', {'title': 'Metrics UP'})

@unauthenticated_user
def user_login(request):
    if request.method == "POST":
        form = UserLoginForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, 'Authorization accepted')
            return redirect('homepage')
        else:
            messages.error(request, 'Authorization failed(Username or password is wrong)')
    else:
        form = UserLoginForm()
    return render(request, 'visit/login.html', {'form': form})


@authenticated_user
def user_logout(request):
    logout(request)
    return redirect('homepage')


@unauthenticated_user
def register(request):
    form = UserRegisterForm(request.POST)
    if request.method == "POST":
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Registration accepted')
            return redirect('login')
        else:
            messages.error(request, 'Failed with registration')
    else:
        form = UserRegisterForm()

    return render(request, 'visit/register.html', {'form': form})


# def user_logout(request):
#     logout(request)
#     return redirect('user_login')
# @method_decorator(allowed_users(['superadmin', 'admin']), name='dispatch')
@adminallow
def adminmain(request):
    return render(request, 'adminpage/main.html', {'title': 'Metrics UP'})


@adminallow
def admtypevehicleslist(request):
    title = "Vehicle types list"
    vehicletypelist = Vehicles.objects.all()
    return render(request, 'adminpage/vehicles/list.html', {
        'title': title,
        'vehicletypelist': vehicletypelist,
    })


@adminallow
def admtypevehicleadd(request):
    title = "Add new type of vehicle"

    if request.method == 'POST':
        vehicletype = VehicletpyesForm(request.POST)
        if vehicletype.is_valid():
            vehicletype = vehicletype.save()
            messages.success(request, "New vehcile type is added")
            return redirect('admtypevehicles', )
        else:
            messages.error(request, "Problem with adding new vehicle type")
    else:
        vehicletype = VehicletpyesForm()
    # return render(request, 'adminpage/vehicles/add.html', {'title': ''})
    return render(request, 'adminpage/vehicles/add.html', {
        'title': title,
        'vehicletype': vehicletype,
    })


class Detailtypevehicle(DetailView, LoginRequiredMixin):
    model = Vehicles
    template_name = 'adminpage/vehicles/detail.html'
    context_object_name = 'vehicletypeinfo'

    # pk_url_kwarg = 'pk'
    def get_object(self):
        id_ = self.kwargs.get('pk')
        owner = get_object_or_404(Vehicles, id=id_)
        return owner
        # if iftapm.truck.division in self.request.user.usersinfo.company.all():
        #     return iftapm
        # else:
        #     raise Http404()

    def get_context_data(self, *, object_list=None, **kwargs):
        # context = super(HomeNews, self).get_context_data()
        context = super().get_context_data(**kwargs)
        context['title'] = "Vehicle type info"
        # context['now'] = datetime.date.today()
        return context


def admtypevehicleupd(request, pk):
    tvehiclemodel = get_object_or_404(Vehicles, pk=pk)

    vehicletype = VehicletpyesForm(request.POST or None, instance=tvehiclemodel)

    if vehicletype.is_valid():
        vehicletype.save()

        messages.success(request, "Vehcile type info is updated")
        return redirect('admtypevehicleview', tvehiclemodel.pk)

    title = "Update vehicle info"
    return render(request, 'adminpage/vehicles/update.html', {
        'title': title,
        'tvehiclemodel': tvehiclemodel,
        'vehicletype': vehicletype,
    })


def admtypevehicledel(request, pk):
    title = "Delete vehicle type info"
    tvehiclemodel = get_object_or_404(Vehicles, pk=pk)

    if request.method == 'POST':
        tvehiclemodel.delete()
        messages.success(request, "Vehcile type info is deleted succesfully")
        return redirect('admtypevehicles')

    return render(request, 'adminpage/vehicles/delete.html', {'title': title, 'tvehiclemodel': tvehiclemodel})


@adminallow
def admrpmsetlist(request):
    title = "RPM setting list"
    rpmsettinglist = Rpmsettings.objects.all()
    return render(request, 'adminpage/rpmset/list.html', {
        'title': title,
        'rpmsettinglist': rpmsettinglist,
    })


def admrpmsetadd(request):
    title = "Add new RPM setting"
    if request.method == 'POST':
        rpmsetting = RpmsettingsForm(request.POST)
        if rpmsetting.is_valid():
            rpmsetting = rpmsetting.save()
            messages.success(request, "New RPM setting is added")
            return redirect('admrpmsetlist', )
        else:
            messages.error(request, "Problem with adding new RPM setting")
    else:
        rpmsetting = RpmsettingsForm()
    return render(request, 'adminpage/rpmset/add.html', {
        'title': title,
        'rpmsetting': rpmsetting,
    })


class Detailrpmset(DetailView, LoginRequiredMixin):
    model = Rpmsettings
    template_name = 'adminpage/rpmset/detail.html'
    context_object_name = 'rpmsetinfo'

    # pk_url_kwarg = 'pk'
    def get_object(self):
        id_ = self.kwargs.get('pk')
        obj = get_object_or_404(Rpmsettings, id=id_)
        return obj

    def get_context_data(self, *, object_list=None, **kwargs):
        # context = super(HomeNews, self).get_context_data()
        context = super().get_context_data(**kwargs)
        context['title'] = "RPM setting info"
        # context['now'] = datetime.date.today()
        return context


def admrpmsetupd(request, pk):
    rpmsettingmodel = get_object_or_404(Rpmsettings, pk=pk)
    rpmsetting = RpmsettingsForm(request.POST or None, instance=rpmsettingmodel)

    if rpmsetting.is_valid():
        rpmsetting.save()

        messages.success(request, "RPM setting info is updated")
        return redirect('admrpmsetview', rpmsettingmodel.pk)

    title = "Update RPM setting info"
    return render(request, 'adminpage/rpmset/update.html', {
        'title': title,
        'rpmsettingmodel': rpmsettingmodel,
        'rpmsetting': rpmsetting,
    })


def admrpmsetdel(request, pk):
    title = "Delete RPM setting info"
    rpmsetmodel = get_object_or_404(Rpmsettings, pk=pk)

    if request.method == 'POST':
        rpmsetmodel.delete()
        messages.success(request, "RPM setting info is deleted succesfully")
        return redirect('admrpmsetlist')

    return render(request, 'adminpage/rpmset/delete.html', {'title': title, 'rpmsetmodel': rpmsetmodel})


@adminallow
def admdisplist(request):
    title = "Dispatchers list"
    displist = Dispatch.objects.all()
    return render(request, 'adminpage/disp/list.html', {
        'title': title,
        'displist': displist,
    })


@adminallow
def admdispadd(request):
    title = "Add new Dispatcher"
    if request.method == 'POST':
        disp = DispatchForm(request.POST)
        if disp.is_valid():
            disp = disp.save()
            messages.success(request, "New Dispatcher is added")
            return redirect('admdisplist', )
        else:
            messages.error(request, "Problem with adding new Dispatcher")
    else:
        disp = DispatchForm()
    return render(request, 'adminpage/disp/add.html', {
        'title': title,
        'disp': disp,
    })


class Detaildisp(DetailView, LoginRequiredMixin):
    model = Dispatch
    template_name = 'adminpage/disp/detail.html'
    context_object_name = 'dispinfo'

    # pk_url_kwarg = 'pk'
    def get_object(self):
        id_ = self.kwargs.get('pk')
        obj = get_object_or_404(Dispatch, id=id_)
        return obj

    def get_context_data(self, *, object_list=None, **kwargs):
        # context = super(HomeNews, self).get_context_data()
        context = super().get_context_data(**kwargs)
        context['title'] = "Dispatcher info"
        # context['now'] = datetime.date.today()
        return context


@adminallow
def admdispupd(request, pk):
    dispmodel = get_object_or_404(Dispatch, pk=pk)
    disp = DispatchForm(request.POST or None, instance=dispmodel)

    if disp.is_valid():
        disp.save()
        messages.success(request, "Dispatcher info is updated")
        return redirect('admdispview', dispmodel.pk)

    title = "Update Dispathcer info"
    return render(request, 'adminpage/disp/update.html', {
        'title': title,
        'dispmodel': dispmodel,
        'disp': disp,
    })


@adminallow
def admdispdel(request, pk):
    title = "Delete Dispatcher info"
    dispmodel = get_object_or_404(Dispatch, pk=pk)

    if request.method == 'POST':
        dispmodel.delete()
        messages.success(request, "Dispatcher info is deleted succesfully")
        return redirect('admdisplist')

    return render(request, 'adminpage/disp/delete.html', {'title': title, 'dispmodel': dispmodel})


@adminallow
def admuserp(request):
    title = "User profile "
    return render(request, 'adminpage/user/userprofile.html', {
        'title': title
    })


@adminallow
def admcompdet(request):
    title = "Company details"
    company = Company.objects.filter(user=request.user)
    if company:
        company = company.first()
    return render(request, 'adminpage/user/company/detail.html', {
        'title': title,
        'company': company
    })


@adminallow
def admcompadd(request):
    title = "Add company details"
    if request.method == 'POST':
        companyinfo = CompanyForm(request.POST)
        if companyinfo.is_valid():
            companyinfo = companyinfo.save(False)
            companyinfo.user = request.user
            companyinfo.save()

            messages.success(request, "Company info is added successfully")
            return redirect('admcompdet', )
        else:
            messages.error(request, "Problem with adding company info")
    else:
        companyinfo = CompanyForm()
    # return render(request, 'adminpage/vehicles/add.html', {'title': ''})
    return render(request, 'adminpage/user/company/add.html', {
        'title': title,
        'companyinfo': companyinfo,
    })


def admcompupd(request, pk):
    companymodel = get_object_or_404(Company, pk=pk)
    companyinfo = CompanyForm(request.POST or None, instance=companymodel)
    if companyinfo.is_valid():
        companyinfo.save()

        messages.success(request, "Comapny info is updated")
        return redirect('admcompdet', )

    title = "Update Company info"
    return render(request, 'adminpage/user/company/update.html', {
        'title': title,
        'companyinfo': companyinfo,
        'companymodel': companymodel,
    })


@adminallow
def admtrucklist(request):
    title = "Trucks list"
    trucklist = Trucks.objects.all()
    return render(request, 'adminpage/truck/list.html', {
        'title': title,
        'trucklist': trucklist,
    })


@adminallow
def admtruckadd(request):
    title = "Add new Truck"
    if request.method == 'POST':
        truck = TrucksForm(request.POST)
        if truck.is_valid():
            truck = truck.save()
            messages.success(request, "New Truck is added")
            return redirect('admtrucklist', )
        else:
            messages.error(request, "Problem with adding new Truck")
    else:
        truck = TrucksForm()
    return render(request, 'adminpage/truck/add.html', {
        'title': title,
        'truck': truck,
    })


class Detailtruck(DetailView, LoginRequiredMixin):
    model = Trucks
    template_name = 'adminpage/truck/detail.html'
    context_object_name = 'truckinfo'

    # pk_url_kwarg = 'pk'
    def get_object(self):
        id_ = self.kwargs.get('pk')
        obj = get_object_or_404(Trucks, id=id_)
        return obj

    def get_context_data(self, *, object_list=None, **kwargs):
        # context = super(HomeNews, self).get_context_data()
        context = super().get_context_data(**kwargs)
        context['title'] = "Truck info"
        # context['now'] = datetime.date.today()
        return context


@adminallow
def admtruckupd(request, pk):
    truckmodel = get_object_or_404(Trucks, pk=pk)
    truck = TrucksForm(request.POST or None, instance=truckmodel)

    if truck.is_valid():
        truck.save()
        messages.success(request, "Truck info is updated")
        return redirect('admtruckview', truckmodel.pk)

    title = "Update Truck info"
    return render(request, 'adminpage/truck/update.html', {
        'title': title,
        'truckmodel': truckmodel,
        'truck': truck,
    })


@adminallow
def admtruckdel(request, pk):
    title = "Delete Truck info"
    truckmodel = get_object_or_404(Trucks, pk=pk)

    if request.method == 'POST':
        truckmodel.delete()
        messages.success(request, "Truck info is deleted succesfully")
        return redirect('admtrucklist')

    return render(request, 'adminpage/truck/delete.html', {'title': title, 'truckmodel': truckmodel})


@adminallow
def admloadslist(request):
    title = "Loads management"
    loads = Loads.objects.all()
    return render(request, 'adminpage/loads/list.html', {
        'title': title,
        'loads': loads,
    })


@adminallow
def admloadsadd(request):
    title = "Add new Load info"
    if request.method == 'POST':
        loads = LoadsForm(request.POST)
        if loads.is_valid():
            loads = loads.save()
            messages.success(request, "New Load is added")
            return redirect('admloadslist', )
        else:
            messages.error(request, "Problem with adding new Load")
    else:
        loads = LoadsForm()
    return render(request, 'adminpage/loads/add.html', {
        'title': title,
        'loads': loads,
    })


class Detailload(DetailView, LoginRequiredMixin):
    model = Loads
    template_name = 'adminpage/loads/detail.html'
    context_object_name = 'loadinfo'

    # pk_url_kwarg = 'pk'
    def get_object(self):
        id_ = self.kwargs.get('pk')
        obj = get_object_or_404(Loads, id=id_)
        return obj

    def get_context_data(self, *, object_list=None, **kwargs):
        # context = super(HomeNews, self).get_context_data()
        context = super().get_context_data(**kwargs)
        context['title'] = "Load info"
        # context['now'] = datetime.date.today()
        return context


@adminallow
def admloadupd(request, pk):
    loadmodel = get_object_or_404(Loads, pk=pk)
    loads = LoadsForm(request.POST or None, instance=loadmodel)

    if loads.is_valid():
        loads.save()
        messages.success(request, "Load info is updated")
        return redirect('admloadview', loadmodel.pk)
    title = "Update Load info"
    return render(request, 'adminpage/loads/update.html', {
        'title': title,
        'loadmodel': loadmodel,
        'loads': loads,
    })


@adminallow
def admloaddel(request, pk):
    title = "Delete Load info"
    loadmodel = get_object_or_404(Loads, pk=pk)

    if request.method == 'POST':
        loadmodel.delete()
        messages.success(request, "Load info is deleted succesfully")
        return redirect('admloadslist')

    return render(request, 'adminpage/loads/delete.html', {'title': title, 'loadmodel': loadmodel})


def forecasting_dashboard(request):
    """Display dashboard for demand forecasting"""
    title = "Demand Forecasting dashboard"
    form = ForecastForm()
    return render(request, 'adminpage/forecasting/dashboard.html', {
        'form': form,
        'title': title
    })


def run_forecast(request):
    """Run the ARIMA forecast on historical data"""
    if request.method == 'POST':
        form = ForecastForm(request.POST)
        if form.is_valid():
            # Data Collection and Preparation
            loads_data = Loads.objects.all().values('pickupdate', 'allmiles')
            data = pd.DataFrame(loads_data)

            # Converting date field to datetime and setting it as index
            data['pickupdate'] = pd.to_datetime(data['pickupdate'])
            data.set_index('pickupdate', inplace=True)

            # Forecasting using ARIMA
            model = ARIMA(data['allmiles'], order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=10)

            # Save Results to Database
            for date, value in zip(pd.date_range(start=data.index[-1], periods=10, freq='D'), forecast):
                ForecastResults.objects.create(forecast_date=date, forecasted_demand=value)

            messages.success(request, 'Forecasting complete')
            return redirect('forecast_results')
    else:
        return redirect('forecasting_dashboard')


def forecast_results(request):
    """Display the results of the forecasting"""
    title = "Forecasting overview"
    results = ForecastResults.objects.all()
    return render(request, 'adminpage/forecasting/results.html', {
        'results': results,
        'title': title
    })


def kpi_clustering_dashboard(request):
    """Dashboard for KPI clustering"""
    title = "KPI Clustering"
    form = KPIClusteringForm()
    return render(request, 'adminpage/kpi/dashboard.html', {
        'form': form,
        'title': title
    })


def run_kpi_clustering(request):
    """Run K-Means clustering on dispatchers and trucks based on KPIs"""
    if request.method == 'POST':
        form = KPIClusteringForm(request.POST)
        if form.is_valid():
            # Data Aggregation
            load_data = Loads.objects.all().values('dispatch_id', 'truck_id', 'totalrate', 'rate')
            dispatch_data = Dispatch.objects.all().values('id', 'fullname')
            truck_data = Trucks.objects.all().values('id', 'unit')

            # Transforming to DataFrames
            load_df = pd.DataFrame(load_data)
            # Merge data based on necessary keys - customized based on desired scheme
            kpi_data = load_df[["totalrate", "rate"]]

            # Clustering
            num_clusters = form.cleaned_data['num_clusters']
            kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(kpi_data)
            load_df['cluster'] = kmeans.labels_

            # Save Results
            for idx, row in load_df.iterrows():
                KPIClusterResults.objects.create(dispatch_id=row['dispatch_id'], truck_id=row['truck_id'],
                                                 cluster_id=row['cluster'])

            messages.success(request, 'KPI Clustering complete')
            return redirect('kpi_clustering_results')
    else:
        return redirect('kpi_clustering_dashboard')


def kpi_clustering_results(request):
    title = "KPI Clustering results"
    """Display results of the KPI clustering"""
    results = KPIClusterResults.objects.select_related('dispatch', 'truck')
    return render(request, 'adminpage/kpi/results.html', {
        'results': results,
        'title': title
    })


def anomaly_detection_dashboard(request):
    """Display dashboard for anomaly detection"""
    title = "Anomaly Detection"
    form = AnomalyDetectionForm()
    return render(request, 'adminpage/anomaly/dashboard.html', {
        'form': form,
        'title': title
    })


def run_anomaly_detection(request):
    """Run Isolation Forest anomaly detection on load data"""
    if request.method == 'POST':
        form = AnomalyDetectionForm(request.POST)
        if form.is_valid():
            # Data Selection and Preprocessing
            load_data = Loads.objects.all().values('totalrate', 'allmiles')
            data = pd.DataFrame(load_data)
            data_scaled = (data - data.mean()) / data.std()  # Normalization

            # Anomaly Detection
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            data['anomaly'] = iso_forest.fit_predict(data_scaled)

            # Save Results
            anomalies = data[data['anomaly'] == -1]
            for index, row in anomalies.iterrows():
                AnomalyResults.objects.create(load=Loads.objects.get(id=index), totalrate=row['totalrate'],
                                              allmiles=row['allmiles'])

            messages.success(request, 'Anomaly Detection complete')
            return redirect('anomaly_detection_results')
    else:
        return redirect('anomaly_detection_dashboard')


def anomaly_detection_results(request):
    """Display results of the anomaly detection"""
    title = "Results Anomaly Detection"
    results = AnomalyResults.objects.select_related('load')
    return render(request, 'adminpage/anomaly/results.html', {
        'results': results,
        'title': title
    })


def visualizations_dashboard(request):
    """Display dashboard for data visualizations"""
    title = "Data Visualization"
    form = VisualizationForm()
    return render(request, 'adminpage/visual/dashboard.html', {
        'form': form,
        'title': title
    })


def render_visualizations(request):
    """Generate visualizations for data analytics"""
    title = "Data Visualization"
    if request.method == 'POST':
        form = VisualizationForm(request.POST)
        if form.is_valid():
            # Data Visualization with Seaborn
            load_data = Loads.objects.all().values('totalrate', 'allmiles', 'pickupdate')
            data = pd.DataFrame(load_data)

            # Generate a static visualization with Seaborn
            plt.figure(figsize=(10, 6))
            sns.barplot(x='pickupdate', y='totalrate', data=data)
            plt.title('Total Rate Over Time')

            # Encode the plot to display it in the template
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode('utf-8').replace('\n', '')

            # Dynamic Visualization with Plotly
            fig = px.line(data, x='pickupdate', y='totalrate', title='Interactive Plotly Visualization')
            plot_div = fig.to_html(full_html=False)

            return render(request, 'adminpage/visual/render.html', {'image_base64': image_base64, 'plot_div': plot_div})
    else:
        form = VisualizationForm()
        return render(request, 'adminpage/visual/dashboard.html', {
            'form': form,
            'title': title
        })
